from pathlib import Path
from load3 import MnistLoadTest

if __name__ == '__main__':
    # std 画像に与えるノイズの分散
    for std in [0.2, 0.4, 0.6]:
                # b-spline transformerの動きの幅
        erase_size = 6
        trans_grid= 3
        for trans_mrange in [0.05, 0.1, 0.2]:
            for cross_id in range(2):
                data_path = Path(
                    f"/home/asanomi/MNISTdata/Mnist_data/train{cross_id}/std{std}-erase{erase_size}-grid{trans_grid}-range{trans_mrange}/img/")
                train_data_loader = MnistLoadTest(data_path)
                img = train_data_loader[0]
                
                test_data_path = Path(f"/home/asanomi/MNISTdata/Mnist_data/test{cross_id}/std{std}-erase{erase_size}-grid{trans_grid}-range{trans_mrange}/img/")
                data_loader = MnistLoadTest(data_path)
                img = data_loader[0]
