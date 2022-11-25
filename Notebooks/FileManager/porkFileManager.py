import pandas as pd
import re

# /pork/(중)경략가격집계 - 소,돼지.csv
def pork_1(
    product_type, 
    target, 
    exclude_cols=['JUDGE_GUBN', 'JUDGE_BREED', 'JUDGE_SEX', 'SABLE_GUBN', 'ABATT_CODE']
):
    product_types = ["돼지 온도체", "돼지 냉도체", "부분육(돼지고기)"]
    targets = ["MAX_COST_AMT", "MIN_COST_AMT"]

    if product_type not in product_types:
        raise ValueError("The product type does not exist!")
    
    if target not in targets:
        raise ValueError("The target is invalid!")

    if product_type == "돼지 온도체":
        JUDGE_KIND = 2
    elif product_type == "돼지 냉도체":
        JUDGE_KIND = 5
    elif product_type == "부분육(돼지고기)":
        JUDGE_KIND = 7

    df = pd.read_csv('../Data/pork/(중)경략가격집계 - 소,돼지.csv', encoding = 'euc_kr', engine ='python')\
            .query(f"JUDGE_KIND == {JUDGE_KIND}")

    df = df.drop(exclude_cols, axis=1)
    df = df.groupby(['STD_DATE']).mean().reset_index()
    df['STD_DATE'] = df['STD_DATE'].apply(lambda x: "20" + "-".join(x.split("/")))
    df.rename(columns={'STD_DATE': 'date'}, inplace=True)

    return df, f"pork({product_type})", f"경락가격({target})"

# /pork/(중)축산유통정보 - 소비자가격.csv
def pork_2(
    product_type=4304, 
    target='DLPC', 
    exclude_cols=[
        'SN', 'INPUT_MTHD_CODE', 'ENTRP_CODE', 'PRDLST_CODE', 'GRAD_CODE', 'GOODS_TPCD', 'GOODS_NM',
        'UNIT', 'HIST_NO', 'PRICE_ERROR_YN', 'LNM', 'USE_YN', 'REGISTER', 'REGISTER_ID', 'UPDUSR', 
        'UPDUSR_ID', 'UPDDE', 'TRN_ID', 'MSG_ID', 'TRN_STATS', 'ERROR_REASON', 'KEYWORD_ERROR_YN', 
        'PCFLT_TIME', 'RGSDE', 'TRN_MSG', 'INSPECT_YN', 'INSPECT_DATE', 'INSPECT_YN2', 'INSPECT_DATE2', 
        'BMS_NO', 'MGR_ABATT_CODE', 'ABATT_CODE', 'POSTN_SPRT_CODE'
    ]
):
    product_types = [4301, 4304, 1, 4401, 4402, 9901, 43, 9902, 6, 4034, 8, 13, 3, 430, 11]
    targets = ['DLPC']
    if target not in targets:
        raise ValueError("Currently, only DLPC is supported!")
    if product_type not in product_types:
        raise ValueError(f"The product {product_type} does not exist!")

    df = pd.read_csv('../Data/pork/(중)축산유통정보 - 소비자가격.csv', encoding = 'euc_kr', engine ='python')
    df = df.drop(exclude_cols, axis=1).query(f"CTSED_CODE == {product_type}") # specific 품종

    df = df.groupby(['TRN_DT']).mean().reset_index()
    df['TRN_DT'] = df['TRN_DT'].apply(lambda x: "20" + "-".join(x.split("/")))
    df.rename(columns={'TRN_DT': 'date'}, inplace=True)

    return df, f"pork(CTSED_CODE={product_type})", "소매가격"

# /pork/축평원_돼지 삼겹살 소매가격.xlsx
def pork_3(
    product_type, 
    target, 
    exclude_cols=[]
):
    product_types = ["삼겹살"]
    targets = ["평균", "최고", "최저"]
    if product_type not in product_types:
        raise ValueError("Only support 삼겹살!")
    if target not in targets:
        raise ValueError("Target must be one of 평균, 최고, 최저!")
    
    df = pd.read_excel(f'../Data/pork/축평원_돼지 삼겹살 소매가격.xlsx', '돼지_삼겹살 소매가격').iloc[1:, :5]
    df.columns = ['year', 'month-day', '평균', '최고', '최저']

    df['year'] = df['year'].interpolate(method='pad')
    df['date'] = df['year'] + " " + df['month-day']
    df['date'] = df['date'].apply(lambda x: '-'.join(re.findall("\d+", x)))

    df = df[['date', '평균', '최고', '최저']]
    df = df.groupby(['date']).mean().reset_index()

    return df, f"pork({product_type})", f"소매가격({target})"

# /pork/축평원_돼지수입 삼겹살 소매가격.xlsx
def pork_4(
    product_type, 
    target, 
    exclude_cols=[]    
):
    product_types = ["수입_돼지고기"]
    targets = ["평균", "최고", "최저"]
    if product_type not in product_types:
        raise ValueError("Only support 수입_돼지고기!")
    if target not in targets:
        raise ValueError("Target must be one of 평균, 최고, 최저!")

    df = pd.read_excel('../Data/pork/축평원_돼지수입 삼겹살 소매가격.xlsx', 'Sheet1').iloc[1:, :5]
    df.columns = ['year', 'month-day', '평균', '최고', '최저']

    df['year'] = df['year'].interpolate(method='pad')
    df['date'] = df['year'] + " " + df['month-day']
    df['date'] = df['date'].apply(lambda x: '-'.join(re.findall("\d+", x)))

    df = df[['date', '평균', '최고', '최저']]
    df = df.groupby(['date']).mean().reset_index()

    return df, f"pork({product_type})", f"소매가격({target})"
