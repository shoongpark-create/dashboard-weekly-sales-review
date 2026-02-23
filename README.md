# Weekly Sales Review Dashboard (GitHub Pages)

이 프로젝트는 주간 세일즈 리뷰 데이터를 HTML 대시보드로 배포합니다.

기본 진입점은 `index.html`이며, 실제 대시보드는 다음 파일입니다.

- `Data/processed/latest/mvp/weekly_sales_mvp_dashboard.html`

## 로컬 실행

정적 서버로 실행하세요.

```bash
python3 -m http.server 8080
```

브라우저 접속:

- `http://localhost:8080/Sherman's%20Workspace/codex_dashboard_weekly_sales_review/`

## 대시보드 재생성

최신 데이터 기반으로 MVP HTML을 다시 만들려면:

```bash
python3 scripts/build_weekly_sales_mvp.py
```

## GitHub Pages 상시 배포

- 워크플로우: `.github/workflows/deploy-pages.yml`
- 트리거: `main`/`master` push 또는 수동 실행
- URL 형식: `https://<github-username>.github.io/<repo-name>/`

초기 1회 설정:

1. 저장소 `Settings > Pages`
2. Source를 `GitHub Actions`로 선택
3. `Actions` 탭에서 배포 워크플로우 성공 확인
