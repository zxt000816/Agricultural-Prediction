import pandas as pd
import re

# 경략가격집계 - 소,돼지.csv
def beef_1(
    product_type, 
    target, 
    exclude_cols=['JUDGE_GUBN', 'JUDGE_BREED', 'JUDGE_SEX', 'SABLE_GUBN', 'ABATT_CODE']    
):
    product_types = ["소", "부분육(쇠고기)"]
    targets = ["MAX_COST_AMT", "MIN_COST_AMT"]

    if product_type not in product_types:
        raise ValueError("The product type does not exist!")
    
    if target not in targets:
        raise ValueError("The target is invalid!")

    if product_type == "소":
        JUDGE_KIND = 1
    elif product_type == "부분육(쇠고기)":
        JUDGE_KIND = 6

    df = pd.read_csv('../Data/beef/경략가격집계 - 소,돼지.csv', encoding = 'euc_kr', engine ='python')\
        .query(f"JUDGE_KIND == {JUDGE_KIND}")

    df = df.drop(exclude_cols, axis=1)
    df = df.groupby(['STD_DATE']).mean().reset_index()
    df['STD_DATE'] = df['STD_DATE'].apply(lambda x: "20" + "-".join(x.split("/")))

    df.rename(columns={'STD_DATE': 'date'}, inplace=True)

    return df, f"beef({product_type})", f"경락가격({target})"

# 축산유통정보 - 소비자가격.csv
def beef_2(
    product_type, 
    target, 
    exclude_cols=[
        'SN', 'INPUT_MTHD_CODE', 'ENTRP_CODE', 'PRDLST_CODE', 'GRAD_CODE', 'GOODS_TPCD', 'GOODS_NM',
        'UNIT', 'HIST_NO', 'PRICE_ERROR_YN', 'LNM', 'USE_YN', 'REGISTER', 'REGISTER_ID', 'UPDUSR', 
        'UPDUSR_ID', 'UPDDE', 'TRN_ID', 'MSG_ID', 'TRN_STATS', 'ERROR_REASON', 'KEYWORD_ERROR_YN', 
        'PCFLT_TIME', 'RGSDE', 'TRN_MSG', 'INSPECT_YN', 'INSPECT_DATE', 'INSPECT_YN2', 'INSPECT_DATE2', 
        'BMS_NO', 'MGR_ABATT_CODE', 'ABATT_CODE', 'POSTN_SPRT_CODE'
    ]  
):
    product_types = [4301, 4401]
    targets = ['DLPC']
    if target not in targets:
        raise ValueError("Currently, only DLPC is supported!")
    if product_type not in product_types:
        raise ValueError(f"The product {product_type} does not exist!")

    df = pd.read_csv('../Data/beef/축산유통정보 - 소비자가격.csv', encoding = 'euc_kr', engine ='python')

    df = df.drop(exclude_cols, axis=1).query(f"CTSED_CODE == {product_type}") # specific 품종

    df = df.groupby(['TRN_DT']).mean().reset_index()
    df['TRN_DT'] = df['TRN_DT'].apply(lambda x: "20" + "-".join(x.split("/")))
    df.rename(columns={'TRN_DT': 'date'}, inplace=True)

    if product_type == 4301:
        product_and_product_type = "beef(CTSED_CODE=4301(소))"
    if product_type == 4401:
        product_and_product_type = "beef(CTSED_CODE=4401(수입 소고기))"
    
    return df, product_and_product_type, "소매가격"


# 축평원_소 수입 소매가격.xlsx
def beef_3(
    product_type, 
    target, 
    exclude_cols=[]  
):
    product_types = ["미국산_갈비", '호주산_갈비', '미국산_갈비살']
    targets = ["평균", "최고", "최저"]
    if product_type not in product_types:
        raise ValueError("This product type does not exist!")
    if target not in targets:
        raise ValueError("Target must be one of 평균, 최고, 최저!")

    df = pd.read_excel('../Data/beef/축평원_소 수입 소매가격.xlsx', product_type).iloc[1:, :5]
    df.columns = ['year', 'month-day', '평균', '최고', '최저']

    df['year'] = df['year'].interpolate(method='pad')
    df['date'] = df['year'] + " " + df['month-day']
    df['date'] = df['date'].apply(lambda x: '-'.join(re.findall("\d+", x)))

    df = df[['date', '평균', '최고', '최저']]
    df = df.groupby(['date']).mean().reset_index()

    return df, f"beef({product_type})", f"소매가격({target})"

# 축평원_한우 소매가격(등심∙설도, 1등급 기준).xlsx
def beef_4(
    product_type, 
    target, 
    exclude_cols=[]     
):
    product_types = ["등심_1+등급", '등심_1등급', '설도_1+등급', '설도_1등급']
    targets = ["평균", "최고", "최저"]
    if product_type not in product_types:
        raise ValueError("This product type does not exist!")
    if target not in targets:
        raise ValueError("Target must be one of 평균, 최고, 최저!")

    df = pd.read_excel('../Data/beef/축평원_한우 소매가격(등심∙설도, 1등급 기준).xlsx', product_type).iloc[1:, :5]

    df.columns = ['year', 'month-day', '평균', '최고', '최저']

    df['year'] = df['year'].interpolate(method='pad')
    df['date'] = df['year'] + " " + df['month-day']
    df['date'] = df['date'].apply(lambda x: '-'.join(re.findall("\d+", x)))

    df = df[['date', '평균', '최고', '최저']]

    df = df.groupby(['date']).mean().reset_index()

    return df, f"beef({product_type})", f"소매가격({target})"