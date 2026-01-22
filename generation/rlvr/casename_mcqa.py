import pandas as pd
import random

df = pd.read_csv("./dataset/raw/lbox-casename/casename.csv")

# 모든 문제 세트 정의 (하드코딩)
problem_sets = [
    # 세트 1
    ["공무집행방해", "공무집행방해, 상해", "공무집행방해, 업무방해", "공무집행방해, 폭행"],
    # 세트 2
    ["업무방해", "폭행", "업무방해, 폭행", "업무방해, 상해"],
    # 세트 3
    ["폭행", "상해", "특수폭행", "특수상해"],
    # 세트 4
    ["마약류관리에관한법률위반(대마)", "마약류관리에관한법률위반(향정)", 
     "마약류관리에관한법률위반(대마), 마약류관리에관한법률위반(향정)", "마약류관리에관한법률위반(마약)"],
    # 세트 5
    ["교통사고처리특례법위반(치상)", "교통사고처리특례법위반(치사)", 
     "교통사고처리특례법위반(치상), 도로교통법위반(음주운전)", "교통사고처리특례법위반(치사), 도로교통법위반(음주운전)"],
    # 세트 6
    ["도로교통법위반(무면허운전)", "도로교통법위반(무면허운전), 도로교통법위반(음주운전)", 
     "도로교통법위반(무면허운전), 도로교통법위반(음주운전), 자동차손해배상보장법위반", 
     "도로교통법위반(음주운전), 자동차손해배상보장법위반"],
    # 세트 7
    ["도로교통법위반(음주운전)", "도로교통법위반(음주운전), 특정범죄가중처벌등에관한법률위반(위험운전치상)",
     "교통사고처리특례법위반(치상), 도로교통법위반(음주운전)", 
     "도로교통법위반(음주운전), 특정범죄가중처벌등에관한법률위반(치상)"],
    # 세트 8 (중복)
    ["교통사고처리특례법위반(치상)", "교통사고처리특례법위반(치사)", 
     "교통사고처리특례법위반(치상), 도로교통법위반(음주운전)", "교통사고처리특례법위반(치사), 도로교통법위반(음주운전)"],
    # 세트 9
    ["사기", "사기, 사문서위조, 위조사문서행사", "사문서위조, 위조사문서행사", "사기, 위조사문서행사"],
    # 세트 10
    ["배상명령신청, 사기", "배상명령신청, 사기, 사문서위조, 위조사문서행사", 
     "배상명령신청, 사기방조", "배상명령신청, 사기, 위조사문서행사"],
    # 세트 11
    ["임금", "근로기준법위반", "근로기준법위반, 근로자퇴직급여보장법위반", "임금, 근로자퇴직급여보장법위반"],
    # 세트 12
    ["성폭력범죄의처벌등에관한특례법위반(카메라등이용촬영)", "성폭력범죄의처벌등에관한특례법위반(카메라등이용촬영·반포등)", "성폭력범죄의처벌등에관한특례법위반(통신매체이용음란)", "아동·청소년의성보호에관한법률위반(음란물소지)"],
    # 세트 13
    ["강제추행", "준강제추행", "강제추행, 주거침입", "준강제추행, 주거침입"],
    # 세트 14
    ["명예훼손", "모욕", "정보통신망이용촉진및정보보호등에관한법률위반(명예훼손)", "명예훼손, 모욕"],
    # 세트 15
    ["절도", "재물손괴", "주거침입, 절도", "주거침입, 재물손괴"],
    # 세트 16
    ["건물명도(인도)", "건물인도", "토지인도", "건물명도(인도), 건물인도"],
    # 세트 17
    ["소유권이전등기", "소유권말소등기", "채무부존재확인", "근저당권말소"],
    # 세트 18
    ["손해배상", "손해배상(산)", "손해배상(의)", "손해배상(자)"],
    # 세트 19
    ["매매대금", "물품대금", "공사대금", "용역비"],
    # 세트 20
    ["대여금", "구상금", "양수금", "추상금"]
]

def standardize(text):
    """띄어쓰기, 특수문자 제거하여 비교"""
    return text.replace(" ", "").replace(",", "").replace("·", "").replace("(", "").replace(")", "").lower()

all_problems = []
total_found = 0
set_results = []

for set_idx, option_set in enumerate(problem_sets, 1):
    print(f"\n=== 세트 {set_idx} 처리 중 ===")
    print(f"선지: {option_set}")
    
    found_casenames = []
    for option in option_set:
        std_option = standardize(option)
        
        matches = df[df['casename'].apply(lambda x: standardize(x) == std_option)]
        
        if len(matches) > 0:
            found_casenames.extend(matches['casename'].unique().tolist())
            print(f"  ✓ '{option}' → {len(matches)}건 찾음")
        else:
            print(f"  ✗ '{option}' → 없음")
    
    found_casenames = list(set(found_casenames))
    
    if found_casenames:

        df_filtered = df[df['casename'].isin(found_casenames)]
        
        for idx, row in df_filtered.iterrows():
            shuffled_options = option_set.copy()
            random.shuffle(shuffled_options)

            correct_idx = None
            row_std = standardize(row['casename'])
            
            for i, opt in enumerate(shuffled_options, 1):
                if standardize(opt) == row_std:
                    correct_idx = i
                    break
            
            problem = {
                'id': row['id'],
                'casename': row['casename'],
                'facts': row['facts'],
                'option_1': shuffled_options[0],
                'option_2': shuffled_options[1],
                'option_3': shuffled_options[2],
                'option_4': shuffled_options[3],
                'correct_answer': f"Answer: {correct_idx}"
            }
            all_problems.append(problem)
        
        total_found += len(df_filtered)
        set_results.append({
            'set': set_idx,
            'found': len(df_filtered),
            'casenames': found_casenames[:3] 
        })
        print(f"  → {len(df_filtered)}개 문제 생성")

df_problems = pd.DataFrame(all_problems)
df_problems.to_csv("./dataset/rlvr/legal_ko/casename.csv", index=False, encoding='utf-8-sig')

# 결과 요약 출력
print(f"\n{'='*60}")
print(f"전체 결과 요약")
print(f"{'='*60}")
print(f"총 {len(problem_sets)}개 세트 처리")
print(f"총 {total_found}개 케이스에서 {len(all_problems)}개 문제 생성")

print(f"\n세트별 결과:")
for result in set_results:
    print(f"  세트 {result['set']}: {result['found']}개 (예: {', '.join(result['casenames'][:2])})")

# 몇 개 예시 출력
if all_problems:
    print(f"\n{'='*60}")
    print("생성된 문제 예시 (처음 3개)")
    print(f"{'='*60}")
    
    for i, prob in enumerate(all_problems[:3], 1):
        print(f"원본 casename: {prob['casename']}")
        print(f"선지:")
        for j in range(1, 5):
            option_text = prob[f'option_{j}']
            if j == prob['correct_answer']:
                print(f"  {['1', '2', '3', '4'][j-1]} {option_text} ← 정답")
            else:
                print(f"  {['1', '2', '3', '4'][j-1]} {option_text}")