# 데이터 구조 분석 스크립트
import pandas as pd
import numpy as np

print("🔍 데이터 구조 재분석...")

# 데이터 로드
try:
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    building_info = pd.read_csv('building_info.csv')
    sample_submission = pd.read_csv('sample_submission.csv')
    
    print("✅ 모든 데이터 파일 로드 성공!")
    
    print("\n=== 📊 TRAIN 데이터 구조 ===")
    print(f"Shape: {train.shape}")
    print(f"Columns: {train.columns.tolist()}")
    
    print("\n=== 📊 TEST 데이터 구조 ===")  
    print(f"Shape: {test.shape}")
    print(f"Columns: {test.columns.tolist()}")
    
    print("\n=== 🏢 BUILDING INFO 구조 ===")
    print(f"Shape: {building_info.shape}")
    print(f"Columns: {building_info.columns.tolist()}")
    
    print("\n=== 📝 SAMPLE SUBMISSION 구조 ===")
    print(f"Shape: {sample_submission.shape}")
    print(f"Columns: {sample_submission.columns.tolist()}")
    
    # 타겟 변수 찾기
    print("\n=== 🎯 타겟 변수 분석 ===")
    target_candidates = [col for col in train.columns if any(keyword in col for keyword in ['전력소비량', 'kWh', 'consumption', 'energy', 'power'])]
    print(f"타겟 후보 컬럼들: {target_candidates}")
    
    # Train과 Test의 차이점 확인
    print("\n=== 🔄 TRAIN vs TEST 차이점 ===")
    train_only = set(train.columns) - set(test.columns)
    test_only = set(test.columns) - set(train.columns)
    common = set(train.columns) & set(test.columns)
    
    print(f"Train에만 있는 컬럼: {list(train_only)}")
    print(f"Test에만 있는 컬럼: {list(test_only)}")
    print(f"공통 컬럼 수: {len(common)}")
    
    # 전력소비량 관련 통계
    if '전력소비량(kWh)' in train.columns:
        print(f"\n=== ⚡ 전력소비량(kWh) 통계 ===")
        target_col = '전력소비량(kWh)'
        print(train[target_col].describe())
        print(f"결측치: {train[target_col].isnull().sum()}")
        print(f"0값 개수: {(train[target_col] == 0).sum()}")
    
except FileNotFoundError as e:
    print(f"❌ 파일을 찾을 수 없습니다: {e}")
    print("현재 작업 디렉토리에 다음 파일들이 있는지 확인하세요:")
    print("- train.csv")
    print("- test.csv") 
    print("- building_info.csv")
    print("- sample_submission.csv")
