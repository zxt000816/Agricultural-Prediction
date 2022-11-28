from .beefFileManager import *
from .porkFileManager import *

# - 소
# 1. ./beef/경략가격집계 - 소,돼지.csv
# 2. ./beef/축산유통정보 - 소비자가격.csv
# 3. ./beef/축평원_소 수입 소매가격.xlsx
# 4. ./beef/축평원_한우 소매가격(등심∙설도, 1등급 기준).xlsx

# - 돼지
# 1. ./pork/경략가격집계 - 소,돼지.csv
# 2. /pork/(중)축산유통정보 - 소비자가격.csv
# 3. /pork/축평원_돼지 삼겹살 소매가격.xlsx
# 4. /pork/축평원_돼지수입 삼겹살 소매가격.xlsx

def dataManager(file_name, product, product_type, target):
    if product == 'beef':
        if file_name == "경략가격집계 - 소,돼지":
            # product type can select from ["소", "부분육(쇠고기)"]
            # target can be select from ["MAX_COST_AMT", "MIN_COST_AMT"]
            df, product_and_product_type, product_attribute = beef_1(
                product_type=product_type, 
                target=target
            )
        if file_name == "축산유통정보 - 소비자가격":
            # product_type can select from [4301, 4401]
            # target must be DLPC
            df, product_and_product_type, product_attribute = beef_2(
                product_type=product_type, 
                target=target
            )

        if file_name == "축평원_소 수입 소매가격":
            # product_type can select from ["미국산_갈비", '호주산_갈비', '미국산_갈비살']
            # target must be ["평균", "최고", "최저"]
            df, product_and_product_type, product_attribute = beef_3(
                product_type=product_type, 
                target=target
            )

        if file_name == "축평원_한우 소매가격(등심∙설도, 1등급 기준)":
            # product_type can select from ["등심_1+등급", '등심_1등급', '설도_1+등급', '설도_1등급']
            # target must be ["평균", "최고", "최저"]
            df, product_and_product_type, product_attribute = beef_4(
                product_type=product_type, 
                target=target
            )

    if product == 'pork':
        if file_name == "(중)경략가격집계 - 소,돼지":
            # product type can select from ["돼지 온도체", "돼지 냉도체", "부분육(돼지고기)"]
            # target can be select from ["MAX_COST_AMT", "MIN_COST_AMT"]
            df, product_and_product_type, product_attribute = pork_1(
                product_type=product_type, 
                target=target
            )

        if file_name == "(중)축산유통정보 - 소비자가격":
            # product_type can select from [4304, 4402]
            # target must be DLPC
            df, product_and_product_type, product_attribute = pork_2(
                product_type=product_type, 
                target=target
            )

        if file_name == "축평원_돼지 삼겹살 소매가격":
            # product type must be 삼겹살
            # product target can select from ["평균", "최고", "최저"]
            df, product_and_product_type, product_attribute = pork_3(
                product_type=product_type, 
                target=target
            )

        if file_name == "축평원_돼지수입 삼겹살 소매가격":
            # product type must be 수입_돼지고기
            # product target can select from ["평균", "최고", "최저"]
            df, product_and_product_type, product_attribute = pork_4(
                product_type=product_type, 
                target=target
            )
    
    return df, product_and_product_type, product_attribute