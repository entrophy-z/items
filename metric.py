# 回测模块
import numpy as np
import polars as pl

# 工具函数
def align_idx_fin_df(*data_dfs):
    codes = []
    dates = []

    for df in data_dfs:
        codes += df['code'].to_list()
        dates += df['date'].to_list()

    code_df = pl.DataFrame({'code': sorted(list(set(codes)))})
    date_df = pl.DataFrame({'date': sorted(list(set(dates)))})
    grid_df = code_df.join(date_df, how='cross')

    return [
        df.join(
            grid_df,
            on=['code', 'date'],
            how='right'
        ).select(['code', 'date'] + [
            col for col in df.columns if col not in ['code', 'date']
        ]) for df in data_dfs
    ]


def df_to_array_3d(data_df):
    # (N, T, D)
    N = data_df['code'].unique().shape[0]
    T = data_df['date'].unique().shape[0]
    D = data_df.shape[1] - 2

    if data_df.shape[0] != N * T:
        data_df = align_idx_fin_df(data_df)[0]

    return data_df.drop(['code', 'date']).to_numpy().reshape((N, T, D))

def array_3d_to_df(data_arr, codes, dates, cols):
    # 默认codes、dates参数与data_arr中的索引顺序一致
    code_df = pl.DataFrame({'code': codes})
    date_df = pl.DataFrame({'date': dates})
    grid_df = code_df.join(date_df, how='cross')

    return pl.concat([
        grid_df,
        pl.from_numpy(data_arr.reshape(
            data_arr.shape[0] * data_arr.shape[1], data_arr.shape[2]
        ), schema=cols)
    ], how='horizontal')

def nanstd(arr, *args, **kwargs):
    _std = np.nanstd(arr, *args, **kwargs)
    return np.where(
        _std == 0,
        1,
        _std
    )


def mask(arr, q):
    if q > 0:
        return arr > np.nanquantile(arr, q, axis=1, keepdims=True)
    else:
        return arr < np.nanquantile(arr, (1 + q), axis=1, keepdims=True)


def rank(arr, axis=0):
    mask = ~np.isnan(arr)
    arr = np.where(
        mask,
        arr,
        np.inf
    )
    return np.where(
        mask,
        np.argsort(
            np.argsort(arr, axis=axis), axis=axis
        ) + 1,
        np.nan
    )

def norm(arr, axis=0, lb=0., ub=1.):
    return (arr - np.nanmin(arr, axis=axis, keepdims=True)) \
        / (np.nanmax(arr, axis=axis, keepdims=True) - np.nanmin(arr, axis=axis, keepdims=True)) \
        * (ub - lb) + lb

def zscore(arr):
    return (arr - np.nanmean(arr, axis=0, keepdims=True)) \
        / nanstd(arr, axis=0, keepdims=True)

def reg(x, y, intercept=True):
    # x:(N, T, D1); y:(N, T, D2)
    x = np.concatenate([
        np.ones([x.shape[0], x.shape[1], 1]), x
    ], axis=2) if intercept else x

    x, y = x.transpose(1, 0, 2), y.transpose(1, 0, 2)

    beta = np.linalg.inv(x.transpose(0, 2, 1) @ x) @ (x.transpose(0, 2, 1) @ y)  # (T, D1, D2)
    resi = (y - x @ beta).transpose(1, 0, 2) # (N, T, D2)

    return beta, resi


# 回测类
class Backtest:
    # 多因子统一回测

    def __init__(self, fct_df, mkt_df):
        fct_df, mkt_df = align_idx_fin_df(fct_df, mkt_df)
        self._codes = mkt_df['code'].unique().to_list()
        self._dates = mkt_df['date'].unique().to_list()
        self._fcts = [col for col in fct_df.columns if col not in ['code', 'date']]
        self._fct_arr = df_to_array_3d(fct_df)  # 多因子 (N, T, D)
        self._mkt_arr = df_to_array_3d(mkt_df)  # 多因子 (N, T, 1)

        # 有效值掩码
        self._fct_mask = ~np.isnan(self._fct_arr)
        self._mkt_mask = ~np.isnan(self._mkt_arr)
        self._align_mask = self._fct_mask & self._mkt_mask

        # 索引值网格
        self._grid_df = pl.DataFrame({'code': self._codes}).join(
            pl.DataFrame({'date': self._dates}), how='cross'
        )

    @staticmethod
    def _mask(arr, q):
        return mask(arr, q)

    @staticmethod
    def _rank(arr):
        return rank(arr)

    @staticmethod
    def _norm(arr):
        return norm(arr)

    @staticmethod
    def _zscore(arr):
        return zscore(arr)

    def cvr(self):
        cvr = np.sum(self._align_mask, axis=0) / np.sum(self._mkt_mask, axis=0)

        return pl.concat([
            pl.DataFrame({'date': self._dates}),
            pl.from_numpy(cvr, schema=self._fcts)
        ], how='horizontal')

    def ic(self, q=0.5, rank=False):
        fct, mkt = np.copy(self._fct_arr), np.copy(self._mkt_arr)
        mkt = np.repeat(mkt, fct.shape[2], axis=2)

        # 生成选券掩码，保证选券为在mkt上为有效
        fct[~self._align_mask] = np.nan
        mask = self._mask(fct, q)

        # 基于选券掩码对齐
        fct[~mask] = np.nan
        mkt[~mask] = np.nan

        # rank下，生成rank_fct, rank_mkt
        if rank:
            fct = self._rank(fct)
            mkt = self._rank(mkt)

        ic = np.nanmean(
            (fct - np.nanmean(fct, axis=0, keepdims=True)) * (mkt - np.nanmean(mkt, axis=0, keepdims=True)),
            axis=0
        ) / (
            nanstd(fct, axis=0) * nanstd(mkt, axis=0)
        )  # [T, D]

        return pl.concat([
            pl.DataFrame({'date': self._dates}),
            pl.from_numpy(ic, schema=self._fcts)
        ], how='horizontal')

    def ret(self, q=0.5, wgt='equal'):
        fct, mkt = np.copy(self._fct_arr), np.copy(self._mkt_arr)
        mkt = np.repeat(mkt, fct.shape[2], axis=2)

        # 生成选券掩码，保证选券为在mkt上为有效
        fct[~self._align_mask] = np.nan
        mask = self._mask(fct, q)

        # 基于选券掩码对齐
        fct[~mask] = np.nan
        mkt[~mask] = np.nan

        # 生成仓位信号
        if wgt == 'equal':
            pos = mask.astype(int) / np.sum(mask, axis=0, keepdims=True)
        elif wgt == 'factor':
            pos = fct / np.nansum(fct, axis=0, keepdims=True)
        else:
            raise ValueError

        ret = np.nanmean(pos * mkt, axis=0)

        return pl.concat([
            pl.DataFrame({'date': self._dates}),
            pl.from_numpy(ret, schema=self._fcts)
        ], how='horizontal')

    def expr(self, expr_fct_df, q=0.5, wgt='equal'):
        expr_fcts = [col for col in expr_fct_df.columns if col not in ['code', 'date']]
        expr_fct = df_to_array_3d(
            expr_fct_df.join(
                self._grid_df,
                on=['code', 'date'],
                how='right'
            )
        ) # (N, T, D1)
        fct = np.copy(self._fct_arr) # (N, T, D2)

        # 生成选券掩码，保证选券为在mkt上为有效
        fct[~self._align_mask] = np.nan
        mask = self._mask(fct, q)

        # 基于选券掩码对齐
        fct[~mask] = np.nan

        # 生成仓位信号
        if wgt == 'equal':
            pos = mask.astype(int) / np.sum(mask, axis=0, keepdims=True)
        elif wgt == 'factor':
            pos = fct / np.nansum(fct, axis=0, keepdims=True)
        else:
            raise ValueError

        expr_fct = expr_fct[..., np.newaxis] # (N, T, D1, 1)
        pos = pos[..., :, np.newaxis] # (N, T, 1, D2)
        expr = np.sum(expr_fct * pos, axis=0)  # (T, D1, D2)

        idx_df = pl.DataFrame({'date': self._dates}).join(
            pl.DataFrame({'expr_fct': expr_fcts}), how='cross'
        )

        return pl.concat([
            idx_df,
            pl.from_numpy(expr.reshape(
                expr.shape[0] * expr.shape[1], expr.shape[2]
            ), schema=self._fcts)
        ], how='horizontal')

    def neu(self, neu_fct_df, intercept=True):
        # 因子暴露，默认expr_fct没有nan
        neu_fcts = [col for col in neu_fct_df.columns if col not in ['code', 'date']]
        neu_fct_arr = df_to_array_3d(
            neu_fct_df.join(
                self._grid_df,
                on=['code', 'date'],
                how='right'
            )
        )

        beta, fct = reg(neu_fct_arr, self._fct_arr, intercept)

        idx_df = pl.DataFrame({'date': self._dates}).join(
            pl.DataFrame({'neu_fct': neu_fcts}), how='cross'
        )

        return pl.concat([
            idx_df,
            pl.from_numpy(beta.reshape(
                beta.shape[0] * beta.shape[1], beta.shape[2]
            ), schema=self._fcts),
        ], how='horizontal'), array_3d_to_df(
            fct, self._codes, self._dates, self._fcts
        )

    def neu_ind(self, ind_fct_df):
        return self.neu(ind_fct_df, intercept=False)

    def neu_style(self, style_fct_df, intercept=True):
        return self.neu(style_fct_df, intercept)

if __name__ == '__main__':
    from datetime import date, timedelta
    # 设置随机种子
    np.random.seed(42)

    # 生成日期范围（最近20个交易日）
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(20)]
    # 生成股票代码（10只股票）
    codes = [f'{i:06d}' for i in range(1, 11)]

    # 1. 因子数据DataFrame (5个因子)
    fct_data = []
    for code in codes:
        for dt in dates:
            row = {
                'code': code,
                'date': dt,
                'momentum': np.random.randn() * 0.5,  # 动量因子
                'value': np.random.randn() * 0.3,  # 价值因子
                'size': np.random.randn() * 0.8,  # 规模因子
                'volatility': np.abs(np.random.randn() * 0.2),  # 波动率因子
                'liquidity': np.random.randn() * 0.4  # 流动性因子
            }
            fct_data.append(row)

    fct_df = pl.DataFrame(fct_data)
    # 随机添加一些缺失值
    fct_df = fct_df.with_columns([
        pl.when(np.random.random() < 0.05).then(None).otherwise(pl.col(col)).alias(col)
        for col in fct_df.columns if col not in ['code', 'date']
    ])

    print("因子数据示例:")
    print(fct_df.head())
    print(f"形状: {fct_df.shape}\n")

    # 2. 市场收益数据 (收益率)
    mkt_data = []
    for code in codes:
        for dt in dates:
            # 收益率，均值0.001，标准差0.02
            ret = np.random.randn() * 0.02 + 0.001
            row = {
                'code': code,
                'date': dt,
                'return': ret
            }
            mkt_data.append(row)

    mkt_df = pl.DataFrame(mkt_data)
    # 随机添加一些缺失值
    mkt_df = mkt_df.with_columns([
        pl.when(np.random.random() < 0.03).then(None).otherwise(pl.col('return')).alias('return')
    ])

    print("市场收益数据示例:")
    print(mkt_df.head())
    print(f"形状: {mkt_df.shape}\n")

    # 3. 行业中性化数据 (行业哑变量)
    industries = ['金融', '科技', '消费', '医药', '制造', '能源']
    ind_data = []
    for code in codes:
        # 每只股票固定一个行业
        ind = np.random.choice(industries)
        for dt in dates:
            row = {
                'code': code,
                'date': dt,
                'industry': ind
            }
            ind_data.append(row)

    ind_df = pl.DataFrame(ind_data)
    # 转换为哑变量
    ind_dummy = []
    for ind in industries:
        ind_dummy.append(
            pl.when(pl.col('industry') == ind).then(1).otherwise(0).alias(f'ind_{ind}')
        )
    ind_df = ind_df.with_columns(ind_dummy).drop('industry')

    print("行业中性化数据示例:")
    print(ind_df.head())
    print(f"形状: {ind_df.shape}\n")

    # 4. 风格因子数据 (用于风格中性化)
    style_data = []
    for code in codes:
        for dt in dates:
            row = {
                'code': code,
                'date': dt,
                'beta': np.random.randn() * 0.5 + 1,  # Beta值
                'size_style': np.random.randn() * 0.8,  # 规模风格
                'value_style': np.random.randn() * 0.6,  # 价值风格
                'momentum_style': np.random.randn() * 0.7  # 动量风格
            }
            style_data.append(row)

    style_df = pl.DataFrame(style_data)

    print("风格因子数据示例:")
    print(style_df.head())
    print(f"形状: {style_df.shape}\n")

    # 5. 表达式因子数据 (用于expr方法)
    expr_data = []
    for code in codes:
        for dt in dates:
            row = {
                'code': code,
                'date': dt,
                'alpha': np.random.randn() * 0.3,  # Alpha信号
                'quality': np.random.randn() * 0.4,  # 质量因子
                'sentiment': np.random.randn() * 0.5  # 情绪因子
            }
            expr_data.append(row)

    expr_df = pl.DataFrame(expr_data)

    b = Backtest(fct_df, mkt_df)
    b.ic()
    b.ret()
    b.expr(ind_df)
    b.expr(style_df)
    b.neu_ind(ind_df)
    b.neu_style(style_df)