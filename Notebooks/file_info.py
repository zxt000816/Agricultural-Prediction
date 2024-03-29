import json, os

product_object = {
    "pork": {
        "(중)경략가격집계 - 소,돼지": {
            "product_types": ["돼지 온도체"],
            "targets": ["MAX_COST_AMT", "MIN_COST_AMT"]
        },
        "(중)축산유통정보 - 소비자가격": {
            "product_types": [4304, 4402],
            "targets": ['DLPC']
        },
        "축평원_돼지 삼겹살 소매가격": {
            "product_types": ["삼겹살"],
            "targets": ["평균", "최고", "최저"]
        },
        "축평원_돼지수입 삼겹살 소매가격": {
            "product_types": ["수입_돼지고기"],
            "targets": ["평균", "최고", "최저"]
        }
    },
    "beef": {
        "경략가격집계 - 소,돼지": {
            "product_types": ["소", "부분육(쇠고기)"],
            "targets": ["MAX_COST_AMT", "MIN_COST_AMT"]
        },
        "축산유통정보 - 소비자가격": {
            "product_types": [4301, 4401],
            "targets": ["DLPC"]
        },
        "축평원_소 수입 소매가격": {
            "product_types": ["미국산_갈비", '호주산_갈비', '미국산_갈비살'],
            "targets": ["평균", "최고", "최저"]
        },
        "축평원_한우 소매가격(등심∙설도, 1등급 기준)": {
            "product_types": ["등심_1+등급", '등심_1등급', '설도_1+등급', '설도_1등급'],
            "targets": ["평균", "최고", "최저"]
        }
    }
}

save_path = "./File information.json"

if not os.path.exists(save_path):
    with open(save_path, 'w', encoding='utf8') as f:
        json.dump(product_object, f, ensure_ascii=False)
else:
    print("Failed to export the file because the file already exists. ")