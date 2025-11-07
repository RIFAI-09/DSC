import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import matplotlib
# use non-GUI backend for automated runs/tests
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


def get_data_paths():
    base = os.path.dirname(__file__)
    data_dir = os.path.join(base, 'fmcg_personalcare')
    return {
        'sales': os.path.join(data_dir, 'sales.csv'),
        'products': os.path.join(data_dir, 'products.csv'),
        'marketing': os.path.join(data_dir, 'marketing.csv')
    }


def read_data(paths):
    sales = pd.read_csv(paths['sales'], parse_dates=['date'])
    products = pd.read_csv(paths['products'], parse_dates=['launch_date'])
    marketing = pd.read_csv(paths['marketing'], parse_dates=['start_date', 'end_date'])
    return sales, products, marketing


def process_data(sales_df, product_df, marketing_df):
    # merge product info
    data_merged = pd.merge(sales_df, product_df, on='product_id', how='left')

    # weekly aggregation using Grouper
    sales_weekly = (
        data_merged
        .groupby(['product_id', pd.Grouper(key='date', freq='W')])
        .agg(
            total_units_sold=('units_sold', 'sum'),
            avg_discount=('discount_pct', 'mean'),
            total_revenue=('revenue', 'sum')
        )
        .reset_index()
        .rename(columns={'date': 'week'})
    )

    # attach product metadata
    sales_weekly = pd.merge(sales_weekly, product_df, on='product_id', how='left')

    # unroll marketing campaigns into weekly spends
    all_weeks = []
    for _, row in marketing_df.iterrows():
        # ensure dates are timestamps
        start = row['start_date']
        end = row['end_date']
        if pd.isna(start) or pd.isna(end) or end < start:
            continue
        campaign_weeks = pd.date_range(start=start, end=end, freq='W')
        if len(campaign_weeks) == 0:
            continue
        spend_per_week = row.get('spend_idr', 0) / len(campaign_weeks)
        for wk in campaign_weeks:
            all_weeks.append({'product_id': row['product_id'], 'week': wk, 'spend_per_week': spend_per_week})

    if all_weeks:
        marketing_unrolled_df = pd.DataFrame(all_weeks)
        marketing_weekly = marketing_unrolled_df.groupby(['product_id', 'week']).agg(
            total_spend=('spend_per_week', 'sum')
        ).reset_index()
    else:
        marketing_weekly = pd.DataFrame(columns=['product_id', 'week', 'total_spend'])

    # merge sales and marketing weekly
    df_final = pd.merge(sales_weekly, marketing_weekly, on=['product_id', 'week'], how='left')
    df_final['total_spend'] = df_final['total_spend'].fillna(0)

    return df_final, sales_weekly, marketing_weekly


def run_analysis(df_final, sales_weekly, product_df, prod_a='PC001', prod_b='PC013'):
    # choose products
    # find launch dates safely
    row_b = product_df[product_df['product_id'] == prod_b]
    row_a = product_df[product_df['product_id'] == prod_a]
    if row_b.empty or row_a.empty:
        raise ValueError(f"Product IDs not found: {prod_a if row_a.empty else ''} {prod_b if row_b.empty else ''}")
    launch_date_B = row_b['launch_date'].iloc[0]
    launch_date_A = row_a['launch_date'].iloc[0]

    # prepare series for chosen product A
    df_korban = df_final[df_final['product_id'] == prod_a].sort_values(by='week')

    # Safe plotting (backend is Agg) - create and save a figure instead of show
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df_korban['week'], df_korban['total_units_sold'], color='tab:blue', label=f'Penjualan {prod_a}')
    ax1.axvline(launch_date_B, color='red', linestyle='--', label=f'Peluncuran {prod_b}')
    ax2 = ax1.twinx()
    ax2.plot(df_korban['week'], df_korban['total_spend'], color='tab:green', linestyle=':', alpha=0.7, label='Marketing Spend')
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), 'analysis_plot.png'))
    plt.close(fig)

    # modeling
    df_model = df_korban.copy()
    df_model['post_launch_B'] = (df_model['week'] >= launch_date_B).astype(int)
    df_model['time_since_start'] = (df_model['week'] - df_model['week'].min()).dt.days
    df_model = df_model.rename(columns={'total_units_sold': 'units', 'avg_discount': 'discount'})
    df_model = df_model.dropna(subset=['units', 'discount', 'total_spend', 'post_launch_B', 'time_since_start'])

    if len(df_model) < 5:
        raise ValueError('Not enough data points for modeling after cleaning.')

    model_formula = 'units ~ discount + total_spend + time_since_start + post_launch_B'
    model = smf.ols(formula=model_formula, data=df_model)
    results = model.fit()
    return results, df_korban


def main():
    paths = get_data_paths()
    sales_df, product_df, marketing_df = read_data(paths)
    df_final, sales_weekly, marketing_weekly = process_data(sales_df, product_df, marketing_df)

    print('sales_weekly shape:', sales_weekly.shape)
    print('marketing_weekly shape:', marketing_weekly.shape)
    print('df_final shape:', df_final.shape)

    # run analysis for default products; wrap in try to show friendly error
    try:
        results, df_korban = run_analysis(df_final, sales_weekly, product_df)
        print(results.summary())
    except Exception as e:
        print('Analysis skipped or failed:', type(e).__name__, e)


if __name__ == '__main__':
    main()