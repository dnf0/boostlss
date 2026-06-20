# CHANGELOG

## v0.1.0 (2026-06-20)

### Chore

* chore: update knowledge graph ([`59648c2`](https://github.com/dnf0/boostlss/commit/59648c2699ae36030c1c4ba93a1f943e3d0f4ad5))

* chore: add dev dependencies for golden tests ([`bcadac2`](https://github.com/dnf0/boostlss/commit/bcadac24ef4e814d3f8004b8ed76ef880b263aaf))

* chore: scaffold boostlss cargo workspace and core crate ([`0c5eb5a`](https://github.com/dnf0/boostlss/commit/0c5eb5ac60d2f03798e885c55fa6ac021041364f))

* chore: scope prettier pre-commit hook to the pre-commit stage

Without an explicit stages key, prettier also ran at the commit-msg
stage, where mirrors-prettier errors on .git/COMMIT_EDITMSG
(&#34;No matching files&#34;, exit 2) and blocks every commit. Restrict it
to the pre-commit stage where file formatting belongs.

Co-Authored-By: Claude Opus 4.8 (1M context) &lt;noreply@anthropic.com&gt; ([`f0c938e`](https://github.com/dnf0/boostlss/commit/f0c938ee9a58fe3f141df9d1cc5b2ff0b752911b))

* chore: complete initial repository setup ([`2f192af`](https://github.com/dnf0/boostlss/commit/2f192afdb420e128d2b089c67c017b358e67f042))

### Ci

* ci: fix semantic release token permissions ([`a9f81fd`](https://github.com/dnf0/boostlss/commit/a9f81fdf57aa8fd8d324878eac52e6a77fb15e63))

* ci: remove invalid hashFiles from job if conditions ([`6dab7c4`](https://github.com/dnf0/boostlss/commit/6dab7c4eae5537dcfe3d83c041c3227fc8d660fa))

* ci: fix pull_request workflow condition evaluation ([`a9e0db6`](https://github.com/dnf0/boostlss/commit/a9e0db6d62fc2f17b3a0bd18faad2992c1943716))

### Documentation

* docs: add stump learner implementation plan ([`062191e`](https://github.com/dnf0/boostlss/commit/062191eb55bd9659d116cde9689035d090957848))

* docs: add stump learner and abstraction refactor design spec ([`d7c2043`](https://github.com/dnf0/boostlss/commit/d7c2043a18b529be336aec959ac48495d1943cd0))

* docs: add cvrisk implementation plan ([`02587ab`](https://github.com/dnf0/boostlss/commit/02587ab844b4ff374d4051b9ed53d1120867e9f6))

* docs: add cvrisk design spec ([`8716feb`](https://github.com/dnf0/boostlss/commit/8716febf86c34d303509169ef7d4ee484d078572))

* docs: credit gamboostlss authors and detail validation strategy ([`78d8578`](https://github.com/dnf0/boostlss/commit/78d857848011681219c8da34618e65c5ad9c8a16))

* docs: add implementation plan for golden tests ([`fa980d5`](https://github.com/dnf0/boostlss/commit/fa980d51c8c6fe89814faa00453a26833090f413))

* docs: add design spec for golden tests and documentation updates ([`1c1f914`](https://github.com/dnf0/boostlss/commit/1c1f91436f6fd5b50b41b490aad2efda302f544d))

* docs: update README with project description ([`9bc464d`](https://github.com/dnf0/boostlss/commit/9bc464dc85718d73b8d11b860a59306663a60482))

* docs(plan): add plan 3 for boosting engine ([`7b09a75`](https://github.com/dnf0/boostlss/commit/7b09a75734f3ef9c334a8cbc29719d8286d092d3))

* docs(plan): add plan 2 for base-learners ([`f993209`](https://github.com/dnf0/boostlss/commit/f9932090721792da96c9665c2d966f2d07153e6f))

* docs(plan): add plan 1 for foundations and families ([`b782941`](https://github.com/dnf0/boostlss/commit/b7829414cf77487cd7ee496e60ac1cae08629440))

* docs: specify safe out-of-range prediction behavior for base-learners

Add section 6.5 defining out-of-support prediction: bols globally
linear by construction; bbs rejects out-of-range x at fit time and uses
linear (boundary-tangent) extrapolation at predict time, matching
mboost (verified against R/bl.R and the 2.5-0 changelog). Adds the
OutOfRange error variant. Makes the &#39;safe linear boundaries for unseen
feature values&#39; property explicit.

Co-Authored-By: Claude Opus 4.8 (1M context) &lt;noreply@anthropic.com&gt; ([`fe4bcc1`](https://github.com/dnf0/boostlss/commit/fe4bcc14adf104e532194387c8ca278d3f10f449))

* docs: add boostlss v1 design spec

Design for an idiomatic Rust reimplementation of gamboostlss (boosting
GAMLSS / distributional regression). Covers the generic-family + enum
base-learner architecture, the four v1 families (Gaussian, Student-t,
Gamma, NBinomial) with exact gradients/links, linear and P-spline
base-learners, cyclical and non-cyclical (inner-loss) fitting, gradient
stabilization, cvrisk tuning, predict/coef, Python bindings, the
four-method testing strategy, and a roadmap. Algorithmic details are
verified against gamboostlss/mboost primary sources.

Co-Authored-By: Claude Opus 4.8 (1M context) &lt;noreply@anthropic.com&gt; ([`a5809c9`](https://github.com/dnf0/boostlss/commit/a5809c9cf3ee529fd245f6fef4b1d52ce08fc91c))

### Feature

* feat: implement stump learner and abstract base learners (#8)

* refactor: abstract learner interface with LearnerFit and LearnerUpdate

* feat: implement stump learner

* feat: expose stump learner to python

---------

Co-authored-by: Daniel Fisher &lt;daniel.fisher@climate-x.com&gt; ([`5be9a0c`](https://github.com/dnf0/boostlss/commit/5be9a0cd73acfd7b5f94d2f0963ea339e0e63bfa))

* feat: complete cvrisk module ([`875f2a4`](https://github.com/dnf0/boostlss/commit/875f2a4457b3d94509ada53071f4a5e1d903c0e6))

* feat: python bindings for cvrisk ([`26ff6db`](https://github.com/dnf0/boostlss/commit/26ff6db3236a111fec1cd63577109ea02b0486a4))

* feat: implement CvRisk runner and result ([`8ad3301`](https://github.com/dnf0/boostlss/commit/8ad3301e8b57f1e1e8cf55c32791a27bb0aa3e19))

* feat: implement cvrisk grid generation and clone bounds ([`385b20c`](https://github.com/dnf0/boostlss/commit/385b20c7372e6f0dc7a267dac2e4b650fedd9517))

* feat: setup cv.rs and Resampling ([`7499376`](https://github.com/dnf0/boostlss/commit/749937695a180e28287fa9a113ba35afb20de58d))

* feat: Add Python Bindings (#6)

* feat: setup boostlss-py crate and maturin pyproject

* feat: expose Family and LinearLearner to Python

* feat: expose BoostLssModel binding with fit and predict methods

* test: add python integration tests for model bindings

* chore: mark task 4 as complete

---------

Co-authored-by: Daniel Fisher &lt;daniel.fisher@climate-x.com&gt; ([`1e2af64`](https://github.com/dnf0/boostlss/commit/1e2af64128a55ed4108c81593cd6464dab674cfe))

* feat: implement cyclical fit loop and prediction (#5)

* chore: add faer dependency for dense linear algebra

* feat: add LearnerFit caching Cholesky factors with faer

* fix: optimize LearnerFit matrix operations and enhance unit test

* chore: track faer test

* fix: address code review feedback for LearnerFit caching and tests

* feat: add Linear base-learner

* feat: add difference penalty matrix and df_to_lambda fallback

* chore: optimizations in linear.rs

* feat: add PSpline base-learner evaluation via Cox-de Boor

* chore: verify BaseLearner implementation passes quality gates

* feat: add engine config and gradient stabilization

* feat: add BoostLss builder API

* feat: add Fitted model structs and predict stub

* feat: add cyclical engine loop stub

* feat: complete cyclical fit loop and prediction

* docs: add plan for Python bindings

---------

Co-authored-by: Daniel Fisher &lt;daniel.fisher@climate-x.com&gt; ([`de0201d`](https://github.com/dnf0/boostlss/commit/de0201d216d2dc641ad1760f88aafbb41ba0cb95))

* feat: export public API in lib.rs ([`1701d89`](https://github.com/dnf0/boostlss/commit/1701d890c120613e3f4c010591e644aeb5663832))

* feat: add NBinomialLss family ([`1a6d717`](https://github.com/dnf0/boostlss/commit/1a6d7177d5622b156bb8afc9fe0c8e3120d6a431))

* feat: add GammaLss family ([`5fad51a`](https://github.com/dnf0/boostlss/commit/5fad51aa2342258bd0c66473eb1a2e5dca63c0c8))

* feat: add StudentTLss family with finite-diff score ([`862b68f`](https://github.com/dnf0/boostlss/commit/862b68f8fdcbf48b549faa33725dc9e80d599c78))

* feat: add GaussianLss family with analytical score ([`5e9b22f`](https://github.com/dnf0/boostlss/commit/5e9b22ff78b4eaff67e50e47e09732c878bc74c6))

* feat: add Family trait with finite-difference ngradient ([`33a0833`](https://github.com/dnf0/boostlss/commit/33a0833a5203b7ebf2b046d85616b5549f62df5a))

* feat: add Link trait and ParamSpec ([`984b196`](https://github.com/dnf0/boostlss/commit/984b1962bbc9358214259f5e663f6bafcc77f895))

* feat: add Dataset struct with validation ([`c0e947c`](https://github.com/dnf0/boostlss/commit/c0e947c0ab5381575505e018b66e3736f20ce82d))

* feat: add typed BoostlssError ([`21a8d6d`](https://github.com/dnf0/boostlss/commit/21a8d6df93ca4cf577ff19a1247f7f56718645ed))

* feat: add weighted-mean/sd and 1-D minimizer utilities ([`8d4c553`](https://github.com/dnf0/boostlss/commit/8d4c553b454c38e9b63fd2759a28d53806cc8469))

### Fix

* fix: extract EPSILON and add nbinomial NLL test ([`cb1bcb0`](https://github.com/dnf0/boostlss/commit/cb1bcb0bc32c7f541520ecfcad52526cb7ea0eb0))

* fix: extract EPSILON and add gamma NLL test ([`fde35d0`](https://github.com/dnf0/boostlss/commit/fde35d0b4b57502db507471e7af0653a5cea4885))

* fix: correct analytical gradient chain rule in GaussianLss ([`8f33a7d`](https://github.com/dnf0/boostlss/commit/8f33a7dce7c0139737b641f7f9e6b091f4441792))

### Refactor

* refactor: enforce Send + Sync for Links and remove stale comment ([`7c94279`](https://github.com/dnf0/boostlss/commit/7c9427919fa62c55079bfc691f4cdab054577d37))

* refactor: encapsulate Dataset fields and provide getters ([`f3b58bd`](https://github.com/dnf0/boostlss/commit/f3b58bdce75379b10a0038b9426254ef8a581418))

### Test

* test: add integration test comparing rust outputs to R gamboostlss ([`028946a`](https://github.com/dnf0/boostlss/commit/028946aa1b4ad1a507a641ca0613b8efcda12ec8))

* test: add gamboostlss fixture generation script ([`f62deb4`](https://github.com/dnf0/boostlss/commit/f62deb4dfc2561f01dbbd1f36f5f85051d07065b))

### Unknown

* Initial commit ([`9b05b05`](https://github.com/dnf0/boostlss/commit/9b05b054a301b4fb475d802771a0941956545952))
