# 주간 매출 대시보드 기획 1차안 (MVP)

## 1) 목적

- W6 기반 주간 매출 데이터로 **실행 의사결정 속도**를 높이는 MVP를 만든다.
- 의사결정 축은 `브랜드 > 채널 > 카테고리 > 아이템 > 매장` 5단 드릴다운으로 통일한다.
- 비교 기준은 `전년 동기간(YY)` + `전주(WoW)`를 동시에 제공한다.

## 2) MVP 범위 (In / Out)

### In Scope

- KPI 3종: `Period TY Sales`, `YoY`, `WoW`
- 우선순위 테이블 5종
  - `brand_decisions.csv`
  - `channel_decisions.csv`
  - `category_decisions.csv`
  - `item_decisions.csv`
  - `store_decisions.csv`
- 실행 로드맵 5단계 (`decision_roadmap.csv`)
- 산출물 3종
  - HTML: `weekly_sales_mvp_dashboard.html`
  - XLSX: `weekly_sales_mvp_dashboard.xlsx`
  - Brief(MD): `weekly_sales_mvp_brief.md`

### Out Scope (MVP 이후)

- 로그인/권한관리
- 실시간 스트리밍 데이터 처리
- 다중 시즌 동시 비교용 시계열 차트 고도화
- 시나리오 시뮬레이션(가격/재고 what-if)

## 3) 현재 데이터/산출 구조 (기준)

- 원천: `Data/Weekly_Sales_Review_W6.xlsx`
- 전처리 최신본: `Data/processed/latest`
- 비교 mart:
  - `Data/processed/latest/marts/compare_period_*.csv`
  - `Data/processed/latest/marts/compare_cumulative_*.csv`
- MVP 출력:
  - `Data/processed/latest/mvp`
  - `mvp_manifest.json`에 파일 경로 및 KPI 기록

## 4) 화면 정보 구조 (MVP IA)

1. **Executive KPI**
   - 당주 순매출, YoY 증감, WoW 증감
2. **Decision Roadmap**
   - 브랜드/채널/카테고리/아이템/매장 순서로 실행 액션 제시
3. **Priority Tables (5개)**
   - 우선순위 점수(`priority_score`) + 행동 라벨(`recommended_action`)
4. **Export/Report**
   - 동일 데이터의 XLSX/MD 제공

## 5) KPI/의사결정 로직 (MVP)

- 기준 매출: `sales_period_ty`
- 비교 컬럼:
  - YoY: `yoy_sales_diff`, `yoy_sales_pct`
  - WoW: `wow_sales_diff`, `wow_sales_pct`
- 우선순위 점수(현행):
  - `sales share * 70 + |yoy| * 20 + |wow| * 10`
- 액션 라벨(현행):
  - `투자 확대 / 긴급 점검 / 단기 회복 액션 / 반등 검증 / 유지/모니터링`

## 6) 단계별 의사결정 게이트 (MVP 운영)

### Gate 1. 기준 매출 확정

- 기본안: `sales_amt_net_v_excl`
- 대안: `sales_amt_net`
- 결정 기준: 재무팀 부가세 기준과 일치 여부

### Gate 2. 우선순위 점수 가중치

- 기본안: `70/20/10`
- 대안: `60/25/15` (변동성 반영 강화)
- 결정 기준: 운영팀의 단기/중기 KPI 비중

### Gate 3. Action Threshold

- 기본안:
  - 투자 확대: `YoY >= 12% and WoW >= 5%`
  - 긴급 점검: `YoY <= -8% and WoW <= -5%`
- 결정 기준: 실제 프로모션/재고 집행 단위와의 일치

### Gate 4. 공개 범위

- 기본안: 브랜드/채널/카테고리까지 외부 공유, 아이템/매장은 내부 전용
- 결정 기준: 민감정보 정책

### Gate 5. 배포 방식

- 기본안: GitHub Pages 정적 배포
- 결정 기준: 갱신 주기, 보안, 운영 인력

## 7) GitHub Pages 배포 전략 (MVP)

> 결정: **A안(Artifact Publish) 채택**

## A안 (권장): Artifact Publish 방식

- 로컬/CI에서 아래 실행 후 산출물 생성
  - `python3 scripts/prepare_weekly_sales_data.py`
- 생성된 `Data/processed/latest/mvp`를 배포 대상으로 사용
- 장점: 정적 산출물 중심으로 배포하여 장애 포인트 감소

### GitHub Pages 설정

- `Settings > Pages`에서 Source를 `GitHub Actions`로 설정
- 워크플로우: `.github/workflows/deploy-pages.yml`
- Publish 대상: `Data/processed/latest/mvp`

## 7.1 시각화 강화 v1 (적용)

- 브랜드 필터 기반 동적 뷰
- 브랜드/채널 매출 Top 막대 시각화
- 브랜드 YoY/WoW 사분면 분포
- 추천 액션 분포 카드
- 우선순위 테이블 5종 동적 렌더링

## B안: CI Build 방식

- GitHub Actions에서 Python 스크립트를 직접 실행하여 산출물 생성
- 장점: 수동 산출물 커밋 불필요
- 리스크: 빌드 환경/의존성(pandas/openpyxl) 관리 필요

## 캐시/운영 정책 (공통)

- HTML/CSV/JSON: 캐시 짧게 (`no-cache` 또는 `max-age=0`)
- XLSX: 다운로드 파일은 `immutable` 정책 선택 가능
- 브랜치 배포: PR 기반 Preview/검증 워크플로우 사용

## 8) 일정 (MVP 1차)

- D1: IA 확정 + GitHub Pages 설정 확정 (A/B안 선택)
- D2: HTML 시각 개선(필터/정렬/검색) + 기준 KPI 고정
- D3: 운영 검증(업데이트-배포 리허설) + 릴리즈

## 9) 완료 기준 (Definition of Done)

- 최신 W6 파일 업데이트 후 1회 실행으로 아래 산출물 자동 재생성
  - HTML, XLSX, MD, CSV, manifest
- GitHub Pages URL에서 대시보드 접근 가능
- 핵심 KPI 3종 + 우선순위 테이블 5종 정상 노출
- 의사결정 로드맵 5단계가 리포트/HTML에 동일하게 반영

## 10) 리스크 및 대응

- **리스크**: 문서/skill 설정에 삭제된 레거시 스크립트 참조 잔존
  - **대응**: 배포 전 참조 정리(운영 문서 기준 경로를 W6 파이프라인으로 통일)
- **리스크**: 데이터 컬럼 타입 혼합(숫자/문자)
  - **대응**: 전처리 단계에서 강제 캐스팅 및 결측 처리 유지
- **리스크**: 배포 후 캐시로 구버전 화면 노출
  - **대응**: 캐시 헤더 정책 적용 + 배포 버전 타임스탬프 노출
