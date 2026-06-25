# early_stopping.py
import numpy as np
import torch


class EarlyStopping:

    def __init__(self, patience=5, verbose=False, path='chekpoint_model.pth'):
        # patience: 最小値の非更新数カウンタ    # verbose: 表示設定    # path: モデルの格納path
        """
        Args: 
            patience (int): 
            verbose (bool): 
            delta (float): 
            path (str): 
            trace_func
        """

        self.patience = patience    # 設定ストップカウンタ
        self.verbose = verbose  # 表示の有無
        self.counter = 0    # 現在のカウンタ値
        self.best_val_loss = None  # ベストスコア
        self.early_stop = False # ストップフラグ
        self.val_loss_min = np.inf  # 前回のベストスコア記録用
        self.path = path    # ベストモデル格納path

    """
    特殊 (call) メソッド
    実際に学習ループ内で最小 loss を更新したか否かを計算させる部分
    """
    def __call__(self, val_loss, model):
        # Check if validation loss is nan
        if np.isnan(val_loss):
            print("Validation loss is NaN. Ignoring this epoch.")
            return

        if self.best_val_loss is None: # 1Epoch 目の処理
            self.best_val_loss = val_loss # 1Epoch 目はそのままベストスコアとして記録
            self.save_chekpoint(val_loss, model) # 記録後にモデルを保存してスコアを表示
        elif val_loss < self.best_loss:   # ベストスコアを更新した場合
            self.best_val_loss = val_loss # ベストスコアを上書き
            self.save_checkpoint(val_loss, model)    # モデルを保存してスコアを表示
            self.counter = 0    # ストップカウンタリセット
        else:   # ベストスコアを更新できなかった場合
            self.counter += 1   # ストップカウンタを +1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  # 現在のカウンタを表示
            if self.patience <= self.counter:   # 設定カウンタを上回ったらストップフラグを True に変更
                self.early_stop = True

    """
    ベストスコア更新時に実行されるチェックポイント関数
    """
    def save_checkpoint(self, val_loss, model):
        if self.verbose:    # 表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)   # ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss    # その時の loss を記録