# Weekly Sales Automation

`Weekly_Sales_Review_W6.xlsx`를 기준으로 CSV 파이프라인을 자동화합니다.

## 1) 1회 실행 (최신 CSV 생성)

```bash
python3 scripts/prepare_weekly_sales_data.py
```

생성 위치:

- `Data/processed/latest/raw/sheet_*.csv`
- `Data/processed/latest/facts/sales_fact_all.csv`
- `Data/processed/latest/marts/agg_*.csv`
- `Data/processed/latest/marts/compare_period_*.csv`
- `Data/processed/latest/marts/compare_cumulative_*.csv`
- `Data/processed/latest/insights/insights_snapshot.json`
- `Data/processed/latest/manifest.json`

MVP 대시보드 산출물도 함께 생성됩니다.

- `Data/processed/latest/mvp/weekly_sales_mvp_dashboard.html`
- `Data/processed/latest/mvp/weekly_sales_mvp_dashboard.xlsx`
- `Data/processed/latest/mvp/weekly_sales_mvp_brief.md`
- `Data/processed/latest/mvp/*_decisions.csv`
- `Data/processed/latest/mvp/mvp_manifest.json`

## 2) 변경 감지 자동 실행

1회 체크:

```bash
python3 scripts/watch_weekly_sales_data.py --once
```

지속 감시:

```bash
python3 scripts/watch_weekly_sales_data.py --interval 10
```

## 3) MVP만 재생성

CSV가 이미 생성된 상태에서 대시보드 MVP 산출물만 다시 만들려면:

```bash
python3 scripts/build_weekly_sales_mvp.py
```

## 4) 업로드 폴더 연동

`Data/uploads/`에 신규 `.xlsx`를 넣으면 최신 파일을 `Data/Weekly_Sales_Review_W6.xlsx`로 동기화 후 재처리합니다.

## 5) 날짜별 스냅샷

처리 시점마다 아래 구조로 보관됩니다.

```text
Data/processed/history/YYYY-MM-DD/HHMMSS/
  latest/
    raw/
    facts/
    marts/
    insights/
    manifest.json
  Weekly_Sales_Review_W6.xlsx
  snapshot_meta.json
```

## 6) GitHub Pages 배포

이 프로젝트는 GitHub Actions + Pages 기준으로 정적 산출물(`Data/processed/latest/mvp`)을 배포합니다.

권장 순서:

1. 최신 산출물 생성

```bash
python3 scripts/prepare_weekly_sales_data.py
```

2. GitHub Pages 설정

- 저장소 `Settings > Pages`에서 Source를 `GitHub Actions`로 설정
- 워크플로우 파일: `.github/workflows/deploy-pages.yml`

3. 접속 경로

- `/` 접속 시 `weekly_sales_mvp_dashboard.html`로 라우팅됩니다.
