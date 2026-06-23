# CHANGELOG

## v0.7.1 (2026-06-23)

### Fix

* fix: resolve PSpline rightmost edge bug, fix python bindings export ([`7451939`](https://github.com/dnf0/boostlss/commit/7451939a55d4337250d7e8f34cdaa5bf029946e3))

### Unknown

* Merge pull request #14 from dnf0/fix/pspline-rightmost-edge

fix: resolve PSpline rightmost edge bug, fix python bindings export ([`9b60a7a`](https://github.com/dnf0/boostlss/commit/9b60a7ad03d94fd34d2a5e54f9dd3212b18095f3))

## v0.7.0 (2026-06-23)

### Documentation

* docs: add implementation plan for cyclic p-splines ([`a899cf8`](https://github.com/dnf0/boostlss/commit/a899cf862d056950c0515cc42fd449b36c18584b))

* docs: add design spec for cyclic p-splines ([`c20e3e8`](https://github.com/dnf0/boostlss/commit/c20e3e8589549c9437ee3849333956db9d8085ba))

### Feature

* feat(python): expose cyclic P-spline parameter to Python API ([`71620f8`](https://github.com/dnf0/boostlss/commit/71620f809eba5d743ed921f0625762e4b097ce61))

* feat(pspline): implement cyclic basis matrix wrapping ([`f20ae75`](https://github.com/dnf0/boostlss/commit/f20ae75d291c08e5f460872825447c980c3bcdc5))

* feat(pspline): add cyclic state and cyclic penalty matrix ([`2a92df2`](https://github.com/dnf0/boostlss/commit/2a92df287b1de44e5dc9dc9ad9e9814ec38898fb))

### Fix

* fix(pspline): correct difference_matrix logic for non-cyclic boundaries and add d=2 tests ([`14aa499`](https://github.com/dnf0/boostlss/commit/14aa4998b7cb43cbfde1e84ffed7bf151c7e085d))

### Unknown

* Merge pull request #13 from dnf0/feat/cyclic-psplines

feat: Cyclic P-splines implementation ([`59927ab`](https://github.com/dnf0/boostlss/commit/59927ab142a37e12c2a16c8279cf49678f1f5f21))

## v0.6.0 (2026-06-23)

### Feature

* feat: implement RandomEffects base-learner (#12)

* docs: add Random Effects learner design spec

* docs: add Random Effects learner implementation plan

* feat: implement RandomEffects base-learner core

* fix(random_effects): add sanity check on n_cols to prevent OOM panic

* feat: implement RandomEffects base-learner core

* feat: implement RandomEffects out-of-bounds predict logic

* feat: expose RandomEffects learner to Python bindings

* chore: rebuild graphify

---------

Co-authored-by: Daniel Fisher &lt;daniel.fisher@climate-x.com&gt; ([`01e30d6`](https://github.com/dnf0/boostlss/commit/01e30d6d7477dd7ddd809bc2fcaba2e9220de234))

## v0.5.0 (2026-06-23)

### Ci

* ci: gracefully skip agent rules update if DEPLOY_TOKEN is missing ([`18c91b0`](https://github.com/dnf0/boostlss/commit/18c91b0c36f2a3b7f29f7a3c7d0ac64f5d82c1b4))

### Feature

* feat: expose Binomial, Beta, Weibull, LogNormal, ZIP, and GEV families to Python ([`a972f16`](https://github.com/dnf0/boostlss/commit/a972f162030f27774ed98ac7218710b4bd46ec4c))

* feat: add GEVLss distribution family ([`d0f6c45`](https://github.com/dnf0/boostlss/commit/d0f6c451a03c4c22c3d01d0782d3bdacfda24fa0))

* feat: add ZIPLss distribution family ([`7dc20ff`](https://github.com/dnf0/boostlss/commit/7dc20ff3561087c960ab6bb9a0870bac7470d642))

* feat: add LogNormalLss distribution family ([`8ce0c72`](https://github.com/dnf0/boostlss/commit/8ce0c72977c7f651a8639b85f882ff061269a0d7))

* feat: add BetaLss distribution family ([`f79f735`](https://github.com/dnf0/boostlss/commit/f79f7358ff7b296934f0c92d6291c1651fdaa076))

* feat: add WeibullLss distribution family ([`cc1d29e`](https://github.com/dnf0/boostlss/commit/cc1d29e6a82072108d634403e3bfd2d0f5a0d34d))

### Fix

* fix: remove allocations from GEVLss init_offsets closures ([`b9e473d`](https://github.com/dnf0/boostlss/commit/b9e473d09ac7402600979341b4469f8a7f848272))

* fix: address code review feedback for LogNormalLss ([`e308c76`](https://github.com/dnf0/boostlss/commit/e308c76f6b4231e4571ea61227969fe2970a1104))

* fix: remove O(N) allocations in BetaLss tight loops ([`28463b0`](https://github.com/dnf0/boostlss/commit/28463b029229b2052910c6239929e0f3115ac4d9))

* fix: remove O(N) allocations in WeibullLss tight loops ([`83b2890`](https://github.com/dnf0/boostlss/commit/83b289010c9605a29f2d80beebf6c28e97b835f2))

### Unknown

* Merge pull request #11 from dnf0/feat/additional-families

feat: Add Beta, Weibull, Log-Normal, ZIP, and GEV distribution families ([`44983dc`](https://github.com/dnf0/boostlss/commit/44983dc06f9b7263738f9120e7bf5e94080d6854))

## v0.4.0 (2026-06-23)

### Chore

* chore: merge main into feature branch ([`3d06613`](https://github.com/dnf0/boostlss/commit/3d0661300415d0f704c38e58e2a512e2d357cb29))

### Ci

* ci: fix github actions and pre-commit config ([`09ce655`](https://github.com/dnf0/boostlss/commit/09ce6558358cf7aea0f02d00ba357acdce588fb5))

* ci: fix unit-test and cargo audit checks ([`a9f6a55`](https://github.com/dnf0/boostlss/commit/a9f6a55cc12fa8a141e083a335755f3ef6d36c3f))

### Documentation

* docs: add implementation plan for BinomialLss ([`95f1566`](https://github.com/dnf0/boostlss/commit/95f15667c912dd309819af200b636dcd1a42be3b))

* docs: add design spec for BinomialLss family ([`ae6a7c7`](https://github.com/dnf0/boostlss/commit/ae6a7c779514e9dbf5badac7874045e26906fb72))

* docs: add implementation plan for feature importance and partial dependence ([`ee5e126`](https://github.com/dnf0/boostlss/commit/ee5e126ba74723de567acfc13a6a3c37c64c5381))

* docs: add design spec for Feature Importance and Partial Dependence ([`cdafea1`](https://github.com/dnf0/boostlss/commit/cdafea1364ab59eb029f2346644a4522444e74b7))

### Feature

* feat: expose BinomialLss to Python bindings ([`b737ea8`](https://github.com/dnf0/boostlss/commit/b737ea8d9316d6521f1b56c1ce217ac89acb07a6))

* feat: implement BinomialLss family for classification ([`2ae1c74`](https://github.com/dnf0/boostlss/commit/2ae1c743e23dcce0ff5621f8d2dec6d94f952d53))

* feat: expose feature_importance and partial_dependence to Python ([`6bc2969`](https://github.com/dnf0/boostlss/commit/6bc2969a308386147d9ebe4ede3b4a1a67ab02af))

* feat: implement partial_dependence computation on Fitted model ([`51df5f7`](https://github.com/dnf0/boostlss/commit/51df5f7d4e8feb102790dc2c0aa22e1c46a381ed))

* feat: implement feature_importance aggregation on Fitted model ([`51ad8bd`](https://github.com/dnf0/boostlss/commit/51ad8bd7efbf3a4713dc0ac9752aba3d88697d0f))

* feat: calculate and store empirical risk reduction in cyclical engine ([`c3cdbf0`](https://github.com/dnf0/boostlss/commit/c3cdbf0e556eaa8b5ad52c517f08351bc9a5ac8d))

* feat: add risk_reduction field to UpdateStep ([`1c8d5c9`](https://github.com/dnf0/boostlss/commit/1c8d5c9d83edb62eb4a0a280094aa64605969319))

### Fix

* fix: refactor FittedModel enum and add missing tests ([`36c4af5`](https://github.com/dnf0/boostlss/commit/36c4af5742ecf6588e952cac699dbe47157881b3))

* fix: address reviewer feedback for binomial lss

- Remove redundant O(N) check_response in BinomialLss::nll
- Vectorize ngradient computation in BinomialLss
- Add comment about LogitLink assumption in analytical gradient simplification ([`3fd5c3b`](https://github.com/dnf0/boostlss/commit/3fd5c3b2d87e8939d54caebc7324a22b678dead9))

* fix: cleaner ndarray conversions and split python tests ([`5e7c279`](https://github.com/dnf0/boostlss/commit/5e7c27938e20c5d31765a23cbba595527fbf027a))

* fix: cleaner ndarray conversions and split python tests ([`4f18c04`](https://github.com/dnf0/boostlss/commit/4f18c04d239bcc69e5edbf5cc7d84c1d60378178))

* fix: address code review feedback for python bindings ([`e520efb`](https://github.com/dnf0/boostlss/commit/e520efb031fe47caf1b2b0b321c9ad56ba7774dc))

* fix: address code review feedback for partial dependence ([`193f230`](https://github.com/dnf0/boostlss/commit/193f2309c1bc3d51858b02b248cf6f676a54b5b1))

### Style

* style: apply pre-commit formatting ([`6cbfb0f`](https://github.com/dnf0/boostlss/commit/6cbfb0f430ca7d28e2b1d74a79da391517d44f21))

### Test

* test: add assertions to test_binomial_fit_predict and move it to a dedicated file

Fixes the issues raised by the code quality reviewer:
1. Adds correctness and boundary assertions to test_binomial_fit_predict.
2. Increases sample size to 200 and mstop to 50 for more reliable learning validation.
3. Moves test_binomial_fit_predict from test_basic.py to test_binomial.py as specified in the implementation plan. ([`237c26c`](https://github.com/dnf0/boostlss/commit/237c26c16e488e696988ecce883f92fee1620099))

### Unknown

* Merge branch &#39;feat/interpretability-serialization&#39;

# Conflicts:
#	.github/workflows/pull_request.yaml ([`557a4cd`](https://github.com/dnf0/boostlss/commit/557a4cde5df06ed127c5acec6c063c03803b17f9))

## v0.3.0 (2026-06-22)

### Feature

* feat: add BinomialLSS family and Python bindings (#10)

* docs: add design spec for Feature Importance and Partial Dependence

* docs: add implementation plan for feature importance and partial dependence

* feat: add risk_reduction field to UpdateStep

* feat: calculate and store empirical risk reduction in cyclical engine

* feat: implement feature_importance aggregation on Fitted model

* feat: implement partial_dependence computation on Fitted model

* fix: address code review feedback for partial dependence

* feat: expose feature_importance and partial_dependence to Python

* fix: address code review feedback for python bindings

* fix: cleaner ndarray conversions and split python tests

* fix: cleaner ndarray conversions and split python tests

* docs: add design spec for BinomialLss family

* docs: add implementation plan for BinomialLss

* feat: implement BinomialLss family for classification

* fix: address reviewer feedback for binomial lss

- Remove redundant O(N) check_response in BinomialLss::nll
- Vectorize ngradient computation in BinomialLss
- Add comment about LogitLink assumption in analytical gradient simplification

* feat: expose BinomialLss to Python bindings

* test: add assertions to test_binomial_fit_predict and move it to a dedicated file

Fixes the issues raised by the code quality reviewer:
1. Adds correctness and boundary assertions to test_binomial_fit_predict.
2. Increases sample size to 200 and mstop to 50 for more reliable learning validation.
3. Moves test_binomial_fit_predict from test_basic.py to test_binomial.py as specified in the implementation plan.

* fix: refactor FittedModel enum and add missing tests

* style: apply pre-commit formatting

* ci: fix unit-test and cargo audit checks

---------

Co-authored-by: Daniel Fisher &lt;daniel.fisher@climate-x.com&gt; ([`092cae5`](https://github.com/dnf0/boostlss/commit/092cae5e5001bce6fa29a09876aee38fac4e1cd5))

## v0.2.0 (2026-06-22)

### Chore

* chore: update Cargo.lock ([`3679320`](https://github.com/dnf0/boostlss/commit/367932059a23de2bfb56eb543b46c33c565268d3))

* chore: add serde, bincode, and serde_json dependencies ([`a2db83f`](https://github.com/dnf0/boostlss/commit/a2db83f50f1710000699063638e027c46e1d2f0b))

* chore: update python bindings for new ParamBuilder API ([`3aaaf94`](https://github.com/dnf0/boostlss/commit/3aaaf94ac3874d42bfd9524abd6758053e7cc5df))

* chore: fix unused import in tree.rs ([`5530180`](https://github.com/dnf0/boostlss/commit/5530180a258ac854d3996edcdb2cd2c5dcd24b27))

### Documentation

* docs: add formula DSL plan ([`cd88a19`](https://github.com/dnf0/boostlss/commit/cd88a19b93b6944c9881d07f1d757b258b645643))

* docs: add formula DSL design spec ([`819bf16`](https://github.com/dnf0/boostlss/commit/819bf163573f86e1b025e68c604a2a993a2cc33f))

* docs: add tree learner design spec ([`c7222f0`](https://github.com/dnf0/boostlss/commit/c7222f0da2542dddd928b9c29dc429cc7a3ca69f))

### Feature

* feat: implement __getstate__ and __setstate__ for Python pickling ([`2b6c3b3`](https://github.com/dnf0/boostlss/commit/2b6c3b38e673c56958b3e2a0ee100606c67c82c6))

* feat: add save and load methods to Fitted model ([`38703b8`](https://github.com/dnf0/boostlss/commit/38703b8e342a14ee6f776fb84d70c13e1258ef04))

* feat: derive Serialize and Deserialize for core models and families ([`f63254c`](https://github.com/dnf0/boostlss/commit/f63254cd6e124896ccca39382655d2392bd49cc8))

* feat: implement ParamBuilder and update BoostLss::on API ([`135daee`](https://github.com/dnf0/boostlss/commit/135daee39a5322c82475179aed3a9791bd1fd857))

* feat: add From impls for base learners to BaseLearner ([`7b51c45`](https://github.com/dnf0/boostlss/commit/7b51c45ef88809f8ecd7c139b1f9a3570f6585c3))

* feat: expose tree learner to python ([`d8210f5`](https://github.com/dnf0/boostlss/commit/d8210f5f9f79b54aa3d5df1c0721a264fadf009b))

* feat: integrate tree predictions and scaling into engine ([`13dfe1d`](https://github.com/dnf0/boostlss/commit/13dfe1daddb8e91d9eb652dad2f5654dc23e0c4d))

* feat: implement recursive tree split search ([`eea7545`](https://github.com/dnf0/boostlss/commit/eea7545ead80f176b937023130c0031fbbff33a3))

* feat: scaffold Tree base learner and update enums ([`4ffb93f`](https://github.com/dnf0/boostlss/commit/4ffb93f2349f4c422fdf10be228380f02b005f7d))

* feat: define tree base learner and treenode structure ([`7af6d6a`](https://github.com/dnf0/boostlss/commit/7af6d6a9f7ce75271dc02b9284e1e36d8e856a90))

### Unknown

* Merge branch &#39;feat/formula-dsl&#39; ([`4933339`](https://github.com/dnf0/boostlss/commit/4933339dc7b7d2a8f8f2d378d1b5fb037adc1331))

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
