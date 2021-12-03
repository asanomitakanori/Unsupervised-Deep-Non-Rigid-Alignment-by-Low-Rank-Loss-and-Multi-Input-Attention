import torch
import torch.nn as nn
import torchvision
from load import MnistLoad



def gen_block(imgs, labels):
    """ cross validationのため，データを３分割
    [0, 1, 2, 3], [4, 5, 6], [7, 8, 9]の３つのブロックに分割
    :param imgs: 画像リスト
    :param labels: ラベルリスト
    :return: blockに分かれたリスト
    """

    blocks = []
    blocks_labels = []
    blocks_labels.append(labels[labels < 5])
    blocks.append(imgs[labels < 5])

    blocks_labels.append(labels[labels > 4])
    blocks.append(imgs[labels > 4])

    return blocks, blocks_labels



if __name__ == '__main__':
    test_im, test_label = torch.load("/home/kazuya/dataset/MNIST_asanomi/MNIST/processed/test.pt")
    train_im, train_label = torch.load("/home/kazuya/dataset/MNIST_asanomi/MNIST/processed/train.pt")
    imgs = torch.cat([test_im, train_im], dim=0)
    labels = torch.cat([test_label, train_label], dim=0)
    batch_size = 8

    blocks, blocks_labels = gen_block(imgs, labels)
    # std 画像に与えるノイズの分散
    for std in [0.2, 0.4, 0.6]:
        # erase_size 欠損の大きさ
        for erase_size in [5, 10, 20]:
            # b-spline transformerのgrid size
            for trans_grid in [3, 5, 7]:
                # b-spline transformerの動きの幅
                for trans_mrange in [0.05, 0.1, 0.2]:
                    for cross_id in range(2):
                        temp_blocks = blocks.copy()
                        temp_blocks_labels = blocks_labels.copy()
                        test_imgs = temp_blocks.pop(cross_id)
                        test_labels = temp_blocks_labels.pop(cross_id)
                        test_dataloader = MnistLoad(test_imgs, batch_size, std=std, erase_size=erase_size,
                                                    trans_grid=trans_grid, trans_mrange=trans_mrange)

                        train_imgs = torch.cat(temp_blocks)
                        train_labels = torch.cat(temp_blocks_labels)
                        train_dataloader = MnistLoad(test_imgs, batch_size, std=std, erase_size=erase_size,
                                                     trans_grid=trans_grid, trans_mrange=trans_mrange)
