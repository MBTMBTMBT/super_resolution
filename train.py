import os
import pickle
import tqdm
import torch
import torch.nn.functional
import math
import tensorflow as tf
from torch.utils.data import DataLoader
from pathlib import Path
from tensorflow import summary
from skimage.metrics import structural_similarity as sk_ssim
from my_utils import get_logger, make_dir, load_checkpoint
from dataset import CroppingDataset
from models import weight_init, \
    ModelSelect, DiscriminatorSelect, SRCNN, SRCNNLinear, ESRGAN, FSRCNN, \
    PatchDiscriminatorSingleInput, PatchDiscriminatorDoubleInput

# check if GPU is available
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")

# STOP STEALING MY GPU MEMORY TENSORFLOW!
gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def mse(slice_a: torch.Tensor, slice_b: torch.Tensor) -> float:
    return torch.mean(torch.square(torch.subtract(slice_a, slice_b))).item()


def psnr(slice_a: torch.Tensor, slice_b: torch.Tensor):
    return 10 * math.log10(1 / mse(slice_a, slice_b))


def train(
        session_name: str,
        output_dir: str,
        model_select: ModelSelect,
        train_dataset_dirs: list,
        val_dataset_dirs: list,
        batch_size_train: int,
        batch_size_val: int,
        img_shape_x: tuple,
        img_shape_y: tuple,
        learning_rate: float,
        epochs: int,
        shuffle: bool,
        num_workers_train: int,
        num_workers_val: int,
        discriminator_select=DiscriminatorSelect.NO_DISCRIMINATOR,
        disc_loss_weight=0.1,
):
    # make dir for output dir
    output_dir = os.path.join(output_dir, session_name)
    make_dir(output_dir)

    log_dir = os.path.join(output_dir, 'logs')
    make_dir(log_dir)

    logger, log_path = get_logger(session_name, log_dir)

    # create tensorboard loggers
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    tb_train_dir = os.path.join(tensorboard_dir, "train")
    tb_val_dir = os.path.join(tensorboard_dir, "val")
    make_dir(tb_train_dir)
    make_dir(tb_val_dir)
    train_summary_writer = summary.create_file_writer(tb_train_dir)
    val_summary_writer = summary.create_file_writer(tb_val_dir)

    train_dataset = CroppingDataset(
        dataset_dir=train_dataset_dirs[0],
        x_size=img_shape_x,
        y_size=img_shape_y,
        resize_mode='random_scale_crop',
        rand_flip=True,
    )
    val_dataset = CroppingDataset(
        dataset_dir=val_dataset_dirs[0],
        x_size=img_shape_x,
        y_size=img_shape_y,
        resize_mode='crop_middle',
        rand_flip=True,
    )
    for i in range(len(train_dataset_dirs)):
        if i == 0:
            continue
        train_dataset.join(
            CroppingDataset(
                dataset_dir=train_dataset_dirs[i],
                x_size=img_shape_x,
                y_size=img_shape_y,
                resize_mode='random_scale_crop',
                rand_flip=True,
            )
        )
    for i in range(len(val_dataset_dirs)):
        if i == 0:
            continue
        val_dataset.join(
            CroppingDataset(
                dataset_dir=val_dataset_dirs[i],
                x_size=img_shape_x,
                y_size=img_shape_y,
                resize_mode='crop_middle',
                rand_flip=False,
            )
        )
    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=shuffle,
        num_workers=num_workers_train,
        batch_size=batch_size_train,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        num_workers=num_workers_val,
        batch_size=batch_size_val,
        prefetch_factor=2,
    )

    # loss functions
    mse_loss_pixel = torch.nn.MSELoss()
    mse_loss_disc = torch.nn.MSELoss()
    mse_loss_gan = torch.nn.MSELoss()

    # get network
    model = None
    if model_select == ModelSelect.SRCNN_SIGMOID:
        model = SRCNN()
    elif model_select == ModelSelect.SRCNN:
        model = SRCNNLinear()
    elif model_select == ModelSelect.FSRCNN:
        model = FSRCNN()
    elif model_select == ModelSelect.ESRGAN:
        model = ESRGAN()
    assert model is not None

    # get_discriminator
    discriminator = None
    if discriminator_select == DiscriminatorSelect.PATCH_DISCRIMINATOR_SINGLE_INPUT:
        discriminator = PatchDiscriminatorSingleInput()
    elif discriminator_select == DiscriminatorSelect.PATCH_DISCRIMINATOR_DOUBLE_INPUT:
        discriminator = PatchDiscriminatorDoubleInput()

    # get optimizers
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )
    disc_optimizer = None
    if discriminator is not None:
        disc_optimizer = torch.optim.Adam(
            discriminator.parameters(), lr=learning_rate / 2
        )

    # if cuda is available, send everything to GPU
    if cuda:
        model = model.cuda()
        mse_loss_pixel = mse_loss_pixel.cuda()
        if discriminator is not None:
            discriminator = discriminator.cuda()
            mse_loss_disc = mse_loss_disc.cuda()

    # use checkpoints
    checkpoint_dir = os.path.join(output_dir, "saved_checkpoints")
    make_dir(checkpoint_dir)

    # get current epoch
    start_epoch = 0
    if Path(log_path).is_file():
        log = open(log_path, 'r')
        next_line = log.readline()
        while next_line:
            if "===EPOCH=FINISH===" in next_line:
                start_epoch += 1
            next_line = log.readline()

    # init networks and optimizers
    # Initialize weights
    if start_epoch != 0:
        # Load pretrained models
        logger.info('Loading previous checkpoint!')
        model, optimizer \
            = load_checkpoint(model, optimizer,
                              os.path.join(checkpoint_dir, "{}_param_{}.pkl".format(
                                  'model', start_epoch - 1)), device=device, logger=logger)
        if discriminator is not None:
            discriminator, disc_optimizer = load_checkpoint(
                discriminator, disc_optimizer,
                os.path.join(checkpoint_dir, "{}_param_{}.pkl".format('disc', start_epoch - 1)),
                device=device, logger=logger)
    else:
        model.apply(weight_init)
        if discriminator is not None:
            discriminator.apply(weight_init)

    # Get the device we're working on.
    logger.debug("cuda & device")
    logger.debug(cuda)
    logger.debug(device)

    # start training
    logger.info("===== start training =====")

    for epoch_idx in range(start_epoch, epochs):
        logger.info("Epoch %d starts" % epoch_idx)
        batch_loss_sum = 0
        disc_batch_loss_sum = 0
        batch_pixel_loss_sum = 0
        disc_batch_loss = None
        batch_pixel_loss = None
        model.train()
        batch_count = 0
        img_x, img_y, out, disc_real, disc_fake, disc_gan = None, None, None, None, None, None
        real_loss, fake_loss, disc_loss, loss_gan, loss_pixel, loss_final = None, None, None, None, None, None
        for img_x, img_y in tqdm.tqdm(train_loader, desc='Ep %d train' % epoch_idx):
            if cuda:
                img_x = img_x.cuda()
                img_y = img_y.cuda()
            out = model(img_x)
            loss_pixel = mse_loss_pixel(out, img_y)
            if discriminator is not None:
                disc_real, disc_fake = None, None
                if discriminator_select == DiscriminatorSelect.PATCH_DISCRIMINATOR_SINGLE_INPUT:
                    disc_real = discriminator(img_y)
                    disc_fake = discriminator(out.detach())
                elif discriminator_select == DiscriminatorSelect.PATCH_DISCRIMINATOR_DOUBLE_INPUT:
                    disc_real = discriminator(img_x, img_y)
                    disc_fake = discriminator(img_x, out.detach())
                real_loss = mse_loss_disc(disc_real, torch.ones_like(disc_real))
                fake_loss = mse_loss_disc(disc_fake, torch.zeros_like(disc_fake))
                disc_loss = (real_loss + fake_loss) * 0.5
                discriminator.zero_grad()
                disc_optimizer.zero_grad()
                disc_loss.backward()
                disc_optimizer.step()
                disc_batch_loss = disc_loss.detach().clone().cpu().item() if cuda \
                    else disc_loss.detach().clone().item()
                disc_batch_loss_sum += disc_batch_loss

                disc_gan = None
                if discriminator_select == DiscriminatorSelect.PATCH_DISCRIMINATOR_SINGLE_INPUT:
                    disc_gan = discriminator(out)
                elif discriminator_select == DiscriminatorSelect.PATCH_DISCRIMINATOR_DOUBLE_INPUT:
                    disc_gan = discriminator(img_x, out)
                loss_gan = mse_loss_gan(disc_gan, torch.ones_like(disc_gan))
                loss_final = loss_gan * disc_loss_weight + loss_pixel * (1 - disc_loss_weight)
                batch_pixel_loss = loss_pixel.detach().clone().cpu().item() if cuda \
                    else loss_pixel.detach().clone().item()
                batch_pixel_loss_sum += batch_pixel_loss
            else:
                loss_final = loss_pixel
            model.zero_grad()
            optimizer.zero_grad()
            loss_final.backward()
            optimizer.step()
            batch_loss = loss_final.detach().clone().cpu().item() if cuda \
                else loss_final.detach().clone().item()
            batch_loss_sum += batch_loss

            # save epoch results to summary
            step = epoch_idx * (len(train_dataset) // batch_size_train) + batch_count
            if len(train_dataset) % batch_size_train != 0:
                step += epoch_idx * len(train_dataset) % batch_size_train
            with train_summary_writer.as_default():
                summary.scalar('batch_loss', batch_loss, step=step)
                if discriminator is not None:
                    summary.scalar('batch_pixel_loss', batch_pixel_loss, step=step)
                    summary.scalar('disc_batch_loss', disc_batch_loss, step=step)
            batch_count += 1

        # These might help saving some memory...
        if cuda:
            img_x, img_y, out = img_x.cpu(), img_y.cpu(), out.cpu()
            if discriminator is not None:
                disc_real, disc_fake, disc_gan = disc_real.cpu(), disc_fake.cpu(), disc_gan.cpu()
                real_loss, fake_loss, disc_loss = real_loss.cpu(), fake_loss.cpu(), disc_loss.cpu()
                loss_gan, loss_pixel, loss_final = loss_gan.cpu(), loss_pixel.cpu(), loss_final.cpu()
            else:
                loss_final = loss_final.cpu()
            torch.cuda.empty_cache()

        batch_loss_mean = batch_loss_sum / batch_count
        logger.info("Epoch %d model loss %f" % (epoch_idx, batch_loss_mean))
        if discriminator is not None:
            batch_pixel_loss_mean = batch_pixel_loss_sum / batch_count
            logger.info("Epoch %d pixel loss %f" % (epoch_idx, batch_pixel_loss_mean))
            disc_batch_loss_mean = disc_batch_loss_sum / batch_count
            logger.info("Epoch %d discriminator loss %f" % (epoch_idx, disc_batch_loss_mean))

        # start validation
        single_img_count = 0
        mse_val_sum = 0
        psnr_val_sum = 0
        ssim_val_sum = 0
        with torch.no_grad():  # cancel gradients to save memory
            model.eval()
            img_x, img_y, out = None, None, None
            for img_x, img_y in tqdm.tqdm(val_loader, desc='Ep %d val' % epoch_idx):
                if cuda:
                    img_x = img_x.cuda()
                    img_y = img_y.cuda()
                out = model(img_x)
                tmp_batch_size = out.shape[0]
                for i in range(tmp_batch_size):
                    single_out_img = torch.clamp(out[i].detach(), min=0, max=1)
                    single_target_img = img_y[i]
                    mse_single_val = mse(single_out_img, single_target_img)
                    psnr_single_val = psnr(single_out_img, single_target_img)
                    single_out_img = torch.squeeze(
                        single_out_img.cpu() if cuda else single_out_img).numpy().transpose((1, 2, 0))
                    single_target_img = torch.squeeze(
                        single_target_img.detach().cpu() if cuda else single_target_img).numpy().transpose((1, 2, 0))
                    ssim_single_val = sk_ssim(single_out_img, single_target_img, channel_axis=2)
                    single_img_count += 1
                    mse_val_sum += mse_single_val
                    psnr_val_sum += psnr_single_val
                    ssim_val_sum += ssim_single_val

            # again, these might help saving some memory...
            if cuda:
                img_x, img_y, out = img_x.cpu(), img_y.cpu(), out.cpu()
                torch.cuda.empty_cache()

        # get mean
        mse_val_sum /= single_img_count
        psnr_val_sum /= single_img_count
        ssim_val_sum /= single_img_count

        # save val results to summary
        with val_summary_writer.as_default():
            summary.scalar('mse', mse_val_sum, step=epoch_idx)
            summary.scalar('psnr', psnr_val_sum, step=epoch_idx)
            summary.scalar('ssim', ssim_val_sum, step=epoch_idx)

        # output to log
        logger.info("epoch %d: mse %f; psnr: %f; ssim: %f"
                    % (epoch_idx, mse_val_sum, psnr_val_sum, ssim_val_sum))

        # save checkpoint
        gen_state_checkpoint = {
            'epoch': epoch_idx,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(gen_state_checkpoint,
                   os.path.join(checkpoint_dir, "{}_param_{}.pkl".format('model', epoch_idx)), pickle_module=pickle)
        if discriminator is not None:
            disc_state_checkpoint = {
                'epoch': epoch_idx,
                'state_dict': discriminator.state_dict(),
                'optimizer': disc_optimizer.state_dict(),
            }
            torch.save(disc_state_checkpoint,
                       os.path.join(checkpoint_dir, "{}_param_{}.pkl".format('disc', epoch_idx)), pickle_module=pickle)

        logger.info("===EPOCH=FINISH===")


def train_on_folds(
        session_name: str,
        output_dir: str,
        x_size: tuple,
        y_size: tuple,
        model_select: ModelSelect,
        discriminator_select: DiscriminatorSelect,
        dataset_dirs: list,
        epochs: int,
        batch_size_train: int,
        batch_size_val: int,
        learning_rate: float,
        num_workers_train: int,
        num_workers_val: int,
        shuffle=True,
        disc_loss_weight=0.1,
        run_tests=None,
):
    if run_tests is None:
        run_tests = len(dataset_dirs)
    for i in range(len(dataset_dirs)):
        if i == run_tests:
            break
        trained = []
        valed = []
        valed.append(dataset_dirs[len(dataset_dirs) - 1 - i])
        for j in range(len(dataset_dirs)):
            if j == len(dataset_dirs) - 1 - i:
                continue
            else:
                trained.append(dataset_dirs[j])
        print(session_name + '-' + str(i))
        train(
            session_name=session_name + '-' + str(i),
            output_dir=output_dir,
            model_select=model_select,
            discriminator_select=discriminator_select,
            train_dataset_dirs=trained,
            val_dataset_dirs=valed,
            batch_size_train=batch_size_train,
            batch_size_val=batch_size_val,
            img_shape_x=x_size,
            img_shape_y=y_size,
            learning_rate=learning_rate,
            disc_loss_weight=disc_loss_weight,
            epochs=epochs,
            shuffle=shuffle,
            num_workers_train=num_workers_train,
            num_workers_val=num_workers_val,
        )


if __name__ == '__main__':
    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++        Experiment on sigmoid SRCNN         +++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    train_on_folds(
        session_name='SRCNN_SIGMOID',
        output_dir=r'E:\my_files\programmes\python\super_resolution_outputs',
        x_size=(240, 240),
        y_size=(480, 480),
        model_select=ModelSelect.SRCNN_SIGMOID,
        discriminator_select=DiscriminatorSelect.NO_DISCRIMINATOR,
        dataset_dirs=[
            r'E:\my_files\programmes\python\super_resolution_images\fold0',
            r'E:\my_files\programmes\python\super_resolution_images\fold1',
            r'E:\my_files\programmes\python\super_resolution_images\fold2',
            r'E:\my_files\programmes\python\super_resolution_images\fold3',
            r'E:\my_files\programmes\python\super_resolution_images\fold4',
        ],
        epochs=60,
        batch_size_train=4,
        batch_size_val=1,
        learning_rate=0.0001,
        num_workers_train=8,
        num_workers_val=8,
        shuffle=True,
        run_tests=5,
    )
    # r'/home/mbt/super_resolution_workplace/super_resolution_images/fold0',
    # r'/home/mbt/super_resolution_workplace/super_resolution_images/fold1',
    # r'/home/mbt/super_resolution_workplace/super_resolution_images/fold2',
    # r'/home/mbt/super_resolution_workplace/super_resolution_images/fold3',
    # r'/home/mbt/super_resolution_workplace/super_resolution_images/fold4',

    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++          Experiment on pure SRCNN          +++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    train_on_folds(
        session_name='SRCNN',
        output_dir=r'E:\my_files\programmes\python\super_resolution_outputs',
        x_size=(240, 240),
        y_size=(480, 480),
        model_select=ModelSelect.SRCNN,
        discriminator_select=DiscriminatorSelect.NO_DISCRIMINATOR,
        dataset_dirs=[
            r'E:\my_files\programmes\python\super_resolution_images\fold0',
            r'E:\my_files\programmes\python\super_resolution_images\fold1',
            r'E:\my_files\programmes\python\super_resolution_images\fold2',
            r'E:\my_files\programmes\python\super_resolution_images\fold3',
            r'E:\my_files\programmes\python\super_resolution_images\fold4',
        ],
        epochs=60,
        batch_size_train=4,
        batch_size_val=1,
        learning_rate=0.0001,
        num_workers_train=8,
        num_workers_val=8,
        shuffle=True,
        run_tests=5,
    )

    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++         Experiment on pure FSRCNN          +++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    train_on_folds(
        session_name='FSRCNN',
        output_dir=r'E:\my_files\programmes\python\super_resolution_outputs',
        x_size=(240, 240),
        y_size=(480, 480),
        model_select=ModelSelect.FSRCNN,
        discriminator_select=DiscriminatorSelect.NO_DISCRIMINATOR,
        dataset_dirs=[
            r'E:\my_files\programmes\python\super_resolution_images\fold0',
            r'E:\my_files\programmes\python\super_resolution_images\fold1',
            r'E:\my_files\programmes\python\super_resolution_images\fold2',
            r'E:\my_files\programmes\python\super_resolution_images\fold3',
            r'E:\my_files\programmes\python\super_resolution_images\fold4',
        ],
        epochs=60,
        batch_size_train=4,
        batch_size_val=1,
        learning_rate=0.0001,
        num_workers_train=8,
        num_workers_val=8,
        shuffle=True,
        run_tests=5,
    )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ Experiment on single-input GAN output SRCNN +++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    train_on_folds(
        session_name='SRCNN-DISC-SINGLE',
        output_dir=r'E:\my_files\programmes\python\super_resolution_outputs',
        x_size=(240, 240),
        y_size=(480, 480),
        model_select=ModelSelect.SRCNN,
        discriminator_select=DiscriminatorSelect.PATCH_DISCRIMINATOR_SINGLE_INPUT,
        dataset_dirs=[
            r'E:\my_files\programmes\python\super_resolution_images\fold0',
            r'E:\my_files\programmes\python\super_resolution_images\fold1',
            r'E:\my_files\programmes\python\super_resolution_images\fold2',
            r'E:\my_files\programmes\python\super_resolution_images\fold3',
            r'E:\my_files\programmes\python\super_resolution_images\fold4',
        ],
        epochs=60,
        batch_size_train=4,
        batch_size_val=1,
        learning_rate=0.0001,
        num_workers_train=8,
        num_workers_val=8,
        shuffle=True,
        disc_loss_weight=0.1,
        run_tests=5,
    )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ Experiment on double-input GAN output SRCNN +++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    train_on_folds(
        session_name='SRCNN-DISC-DOUBLE',
        output_dir=r'E:\my_files\programmes\python\super_resolution_outputs',
        x_size=(240, 240),
        y_size=(480, 480),
        model_select=ModelSelect.SRCNN,
        discriminator_select=DiscriminatorSelect.PATCH_DISCRIMINATOR_DOUBLE_INPUT,
        dataset_dirs=[
            r'E:\my_files\programmes\python\super_resolution_images\fold0',
            r'E:\my_files\programmes\python\super_resolution_images\fold1',
            r'E:\my_files\programmes\python\super_resolution_images\fold2',
            r'E:\my_files\programmes\python\super_resolution_images\fold3',
            r'E:\my_files\programmes\python\super_resolution_images\fold4',
        ],
        epochs=60,
        batch_size_train=4,
        batch_size_val=1,
        learning_rate=0.0001,
        num_workers_train=8,
        num_workers_val=8,
        shuffle=True,
        disc_loss_weight=0.1,
        run_tests=5,
    )

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ Experiment on single-input GAN output FSRCNN +++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++
    train_on_folds(
        session_name='FSRCNN-DISC-SINGLE',
        output_dir=r'E:\my_files\programmes\python\super_resolution_outputs',
        x_size=(240, 240),
        y_size=(480, 480),
        model_select=ModelSelect.FSRCNN,
        discriminator_select=DiscriminatorSelect.PATCH_DISCRIMINATOR_SINGLE_INPUT,
        dataset_dirs=[
            r'E:\my_files\programmes\python\super_resolution_images\fold0',
            r'E:\my_files\programmes\python\super_resolution_images\fold1',
            r'E:\my_files\programmes\python\super_resolution_images\fold2',
            r'E:\my_files\programmes\python\super_resolution_images\fold3',
            r'E:\my_files\programmes\python\super_resolution_images\fold4',
        ],
        epochs=60,
        batch_size_train=4,
        batch_size_val=1,
        learning_rate=0.0001,
        num_workers_train=8,
        num_workers_val=8,
        shuffle=True,
        disc_loss_weight=0.1,
        run_tests=5,
    )

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ Experiment on double-input GAN output FSRCNN +++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++
    train_on_folds(
        session_name='FSRCNN-DISC-DOUBLE',
        output_dir=r'E:\my_files\programmes\python\super_resolution_outputs',
        x_size=(240, 240),
        y_size=(480, 480),
        model_select=ModelSelect.FSRCNN,
        discriminator_select=DiscriminatorSelect.PATCH_DISCRIMINATOR_DOUBLE_INPUT,
        dataset_dirs=[
            r'E:\my_files\programmes\python\super_resolution_images\fold0',
            r'E:\my_files\programmes\python\super_resolution_images\fold1',
            r'E:\my_files\programmes\python\super_resolution_images\fold2',
            r'E:\my_files\programmes\python\super_resolution_images\fold3',
            r'E:\my_files\programmes\python\super_resolution_images\fold4',
        ],
        epochs=60,
        batch_size_train=4,
        batch_size_val=1,
        learning_rate=0.0001,
        num_workers_train=8,
        num_workers_val=8,
        shuffle=True,
        disc_loss_weight=0.1,
        run_tests=5,
    )
