# Real tables added for claim C14

The refusal measurement (`benchmarks/evidence/refusal.py`) reads these
public OpenML tables from scikit-learn's local cache
(`sklearn.datasets.get_data_home()/openml`), not from this folder. They were
fetched once with `sklearn.datasets.fetch_openml(data_id=ID)` from
https://www.openml.org (no credentials); the runs then read them offline,
and `refusal.fetch` refuses a table whose ARFF file (decompressed, as
OpenML serves it) does not have the SHA-256 in `refusal.PINS`.

The second round of tables (from mfeat-factors on) was chosen by the rule
stated in the module docstring of `refusal.py`, fixed before any of them
was run: every dataset of OpenML-CC18 (study 99), the AutoML benchmark
classification suite (271), OpenML-CTR23 (353) and the AutoML benchmark
regression suite (269), less those already listed, copies of tables
another evidence set uses (`refusal.COPIES`), tables over `refusal.CELLS`
cells, tables with no stated licence, copies of a table of the sealed
TabArena-v0.1 hold-out (`refusal.SEALED`: credit-g, diabetes, diamonds,
Amazon_employee_access, Diabetes130US, KDDCup09_appetency,
physiochemical_protein and QSAR-TID-11 were removed by this rule) and
datasets OpenML has deactivated.

With the hold-out copies removed, custody gave validation and acceptance
fewer than the 59 real tasks each that the rate needs, and the four suites
held no further admissible table. The third round (`refusal.FURTHER`, from
soybean on) takes the datasets of OpenML100 (study 14) in the study's order
under the same rule until custody gives each of them 59; it stopped at
profb. Those tables are not screened by the leakage set. Licences are as OpenML states
them; the socmob table (44987) is for non-commercial scholarly use, and
health_insurance (44993) and black_friday (41540) are GPL.

| OpenML id | name | version | licence | target | declared task | ARFF SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | kr-vs-kp | 1 | Public | `class` | classification | see `PINS` |
| 11 | balance-scale | 1 | Public | `class` | classification | see `PINS` |
| 15 | breast-w | 1 | Public | `Class` | classification | see `PINS` |
| 23 | cmc | 1 | Public | `Contraceptive_method_used` | classification | see `PINS` |
| 50 | tic-tac-toe | 1 | Public | `Class` | classification | see `PINS` |
| 53 | heart-statlog | 1 | Public | `class` | classification | see `PINS` |
| 59 | ionosphere | 1 | Public | `class` | classification | see `PINS` |
| 61 | iris | 1 | Public | `class` | classification | see `PINS` |
| 188 | eucalyptus | 1 | Public | `Utility` | classification | see `PINS` |
| 307 | vowel | 2 | Public | `Class` | classification | see `PINS` |
| 333 | monks-problems-1 | 1 | Public | `class` | classification | see `PINS` |
| 1462 | banknote-authentication | 1 | Public | `Class` | classification | see `PINS` |
| 1480 | ilpd | 1 | Public | `Class` | classification | see `PINS` |
| 1494 | qsar-biodeg | 1 | Public | `Class` | classification | see `PINS` |
| 1049 | pc4 | 1 | Public | `c` | classification | see `PINS` |
| 1063 | kc2 | 1 | Public | `problems` | classification | see `PINS` |
| 1068 | pc1 | 1 | Public | `defects` | classification | see `PINS` |
| 40982 | steel-plates-fault | 3 | Public | `target` | classification | see `PINS` |
| 40983 | wilt | 2 | Public | `class` | classification | see `PINS` |
| 40994 | climate-model-simulation-crashes | 4 | Public | `outcome` | classification | see `PINS` |
| 1487 | ozone-level-8hr | 1 | Public | `Class` | classification | see `PINS` |
| 197 | cpu_act | 1 | Public | `usr` | regression | see `PINS` |
| 216 | elevators | 1 | Public | `Goal` | regression | see `PINS` |
| 218 | house_8L | 1 | Public | `price` | regression | see `PINS` |
| 507 | space_ga | 1 | Public | `ln(VOTES/POP)` | regression | see `PINS` |
| 503 | wind | 1 | Public | `MAL` | regression | see `PINS` |
| 196 | autoMpg | 1 | Public | `class` | regression | see `PINS` |
| 230 | machine_cpu | 1 | Public | `class` | regression | see `PINS` |
| 315 | us_crime | 1 | Public | `ViolentCrimesPerPop` | regression | see `PINS` |
| 41021 | Moneyball | 2 | Public | `RS` | regression | see `PINS` |
| 560 | bodyfat | 1 | Public | `class` | regression | see `PINS` |
| 12 | mfeat-factors | 1 | Public | `class` | classification | `609d8f01dbf3f34a953617cc3586ddbb897d98013eba4ef4a4bf73791205ae5c` |
| 32 | pendigits | 1 | Public | `class` | classification | `5be0ce5f618a248a15e953bb63bd2c6bd99c1b9cdcd59c0a6abf06ae58758bca` |
| 182 | satimage | 1 | Public | `class` | classification | `85a02ca850ecd8a005f746fa4d4dfe543250d8da51ef90590ba9db0f31546850` |
| 38 | sick | 1 | Public | `Class` | classification | `85c39615380ca4c05c17a53c3faad9523924958d1e9bc5af0dcbefc6339e7800` |
| 300 | isolet | 1 | Public | `class` | classification | `b18b12257017ab9fac7015f93cfefaee0291b28c6bcbdc11520c0ac2aa1a9ee2` |
| 458 | analcatdata_authorship | 1 | Public | `Author` | classification | `8674a81705440cfcb458e481300185293c1fe75576c2d86d426b0be8148d4506` |
| 469 | analcatdata_dmft | 1 | Public | `Prevention` | classification | `d1fce1e52ef3e29c3cb8af6fc8215dd2982f829c0bfcd18a8be39269db4df501` |
| 1050 | pc3 | 1 | Public | `c` | classification | `2ef1d5e4712544e44213a23532cf5fd8e8515ad22fa428b9f3443e4190a8e57e` |
| 1053 | jm1 | 1 | Public | `defects` | classification | `8f3795e84e8755f1b21b4820c9f3286e60843d3f0a9751c595e0c47e786a7b52` |
| 4134 | Bioresponse | 1 | Public | `target` | classification | `b8fbe587f40c8f48d1f30003b0b40f085cd32997f7167898228178c5a0919d92` |
| 1497 | wall-robot-navigation | 1 | Public | `Class` | classification | `a7facd7720353055e1ef0af024448e002c551d0fd5defe0fc651661b7c187b3b` |
| 1501 | semeion | 1 | Public | `Class` | classification | `3a0bfdb44b8a11647d9ada08ceadac8523b220d880aff266d7321e605576246e` |
| 1485 | madelon | 1 | Public | `Class` | classification | `ac33b79e8849b0e2e0417c62c50a421237b3dd522c2b344f8da3c0f72a80ad7c` |
| 1486 | nomao | 1 | Public | `Class` | classification | `94b4e6229e434d04fb61f081e547c95ec0bfa99e9ea9e880acd5276d13b972e6` |
| 1468 | cnae-9 | 1 | Public | `Class` | classification | `5bff41d9dc6ab5a11059d4c9fbb09564739d2f98f836b6b90005ab8c7e45b420` |
| 1475 | first-order-theorem-proving | 1 | Public | `Class` | classification | `8ada14230e4df0a3eb12aafd9af2bff99ee83bbb539c97d4fef169d981ca8c51` |
| 4534 | PhishingWebsites | 1 | Public | `Result` | classification | `3bcf93b1612306537ee2737751f1bac185b2a77886ab376644c3bf0c7f7cb3e6` |
| 6332 | cylinder-bands | 2 | Public | `band_type` | classification | `6e792f09b2d59d0500a2ba9a5975b31b5eef9534b98f5bf87ab8047452e4c45e` |
| 4538 | GesturePhaseSegmentationProcessed | 1 | Public | `Phase` | classification | `939a5a0dda808213908be1f2385fc1bbb13e9df68ca7d64d1466f2da3f9824e2` |
| 1478 | har | 1 | Public | `Class` | classification | `16e55052cbf8a3f9d7bc769501e4cb1c7909f50fd3947b8457f5a8c1f75fef4b` |
| 23381 | dresses-sales | 2 | Public | `Class` | classification | `44057bb677eb4a144d01816262fc164f923a46fdda019a026e83e0c44b58ee95` |
| 40499 | texture | 1 | Public | `Class` | classification | `9143690415c61a8d9c33ededf947126fa71cced8f91d50642f06621f84e3e9bd` |
| 40668 | connect-4 | 2 | public | `class` | classification | `430cc660591a281550b4270886776a7c0a7f8047bbdbbcc6ee428f223369e219` |
| 40966 | MiceProtein | 4 | Public | `class` | classification | `f48ced98ecfe8ff056d8ea3507e067258c82f29199a1af2261b516010827cc4d` |
| 41027 | jungle_chess_2pcs_raw_endgame_complete | 1 | Public | `class` | classification | `647f9cded1b34558bd4587106bfabcb908e706194e78d3014fea6d594e909d75` |
| 23517 | numerai28.6 | 2 | Public | `attribute_21` | classification | `df837470119d913b9291342efe4b19abcb302c80bb86b5e372ededc4591b9d14` |
| 40978 | Internet-Advertisements | 2 | Public | `class` | classification | `853b895fb12183ee665de29559d8aab1dc77abcf24602e4c3132b23b035b9a09` |
| 44958 | auction_verification | 7 | CC BY 4.0 | `verification.time` | regression | `f0dcaf7165b3235a6031395f2068bb7833bfbeb32c263713cf8834bd679a4d97` |
| 44964 | superconductivity | 7 | CC BY 4.0 | `critical_temp` | regression | `fdd7d82630afd1b35718bcd1772403db6068d85ec4a1cd8b8dc3b9ca0e0440c8` |
| 44965 | geographical_origin_of_music | 7 | CC BY 4.0 | `latitude` | regression | `56fda911ab67dbccc093345d5e8e112ee40186899c20a8dc8f29bd15b063adb8` |
| 44966 | solar_flare | 7 | CC BY 4.0 | `c_class_flares` | regression | `34f377d0c949b0a35e7b6ce5fd9854f4154842dd03e46d24cfcebe2d5c267794` |
| 44969 | naval_propulsion_plant | 7 | CC BY 4.0 | `gt_compressor_decay_state_coefficient` | regression | `566a503cb6bcd1166f79346cadef9101a4be46573471b80158a6419fd0deea58` |
| 44973 | grid_stability | 7 | CC BY 4.0 | `stab` | regression | `3e3e746b09ed94fac41732fe72912302c0b0a7b9ac683abf0db721299007d6a4` |
| 44974 | video_transcoding | 6 | CC BY 4.0 | `utime` | regression | `4842dde3ca6ad962c613e02b7b2627ba70e6cdd078a8be885b52d6ce8e3d49de` |
| 44975 | wave_energy | 6 | CC BY 4.0 | `energy_total` | regression | `e16a2d7bb0b8d1594816aa44474cd1a6c4750498afe3ebe29d51fa5744144410` |
| 44976 | sarcos | 9 | Public | `V22` | regression | `737ec889f0b6622917746d05f9737fad1c3cfb239114994ec65f718d79774d52` |
| 44981 | pumadyn32nh | 2 | Public | `thetadd6` | regression | `0756f4da4500b61e2025862e2b550ee83fc8684616b1a77cebf6dccd5e13d705` |
| 44984 | cps88wages | 2 | Public | `wage` | regression | `2bbaec07b4fc663edfbe3c3a0ef0b300516043195f36f5d1f69c38a8e9733306` |
| 44987 | socmob | 4 | Non-commercial scholarly and teaching purposes. | `counts_for_sons_current_occupation` | regression | `299f17112726f2f67e3802ec99a3847930ff4e1c1cc6150736afbcbda406f8ac` |
| 44990 | brazilian_houses | 16 | CC 0: Public Domain | `total` | regression | `094759fa443ddbcf885910e4a0fe291fdda5f226d30bc67df29840bef8781619` |
| 44992 | fps_benchmark | 2 | CC BY | `FPS` | regression | `0a4488f0e01971e5a3bf8eaf07848da90b9acc555ddd7278c111ce4150017e38` |
| 44993 | health_insurance | 3 | GPL (>= 2) | `whrswk` | regression | `f97193f812b47d30dc3ae87cd30fab5c468ef04893e7792933df873340514605` |
| 45012 | fifa | 19 | CC0: Public Domain | `wage_eur` | regression | `00b905455c0717e88a08e3e2021437889fd6a91d869b936cb1f46fc66f6e8297` |
| 44962 | forest_fires | 9 | CC BY 4.0 | `area` | regression | `25672c7dd97b7af4c00b5d5812c17f6fde8a9e422835e3610c730117c8d6e2b5` |
| 44967 | student_performance_por | 8 | CC BY 4.0 | `G3` | regression | `ab4db935dca19424413e68129cab61609d6340d83194f185bb4e74e2268e4bd6` |
| 44994 | cars | 4 | Public | `Price` | regression | `98a42023fa9701a1266dcea95b0808807f33b1f346d24b5503d89048656ae48c` |
| 181 | yeast | 1 | Public | `class_protein_localization` | classification | `ae97eccd3ad5f9a998efec8acb2026f1748df6c8a392a5b2b0f321c2ef053933` |
| 1457 | amazon-commerce-reviews | 1 | Public | `Class` | classification | `b5388cae01347259e7f8801c4220022f0d1349d88dd3e8ff9d0c21d2e3c23438` |
| 41138 | APSFailure | 1 | CC0 | `class` | classification | `e95128fc82ef5d5bc5a732b0bc63dea0f6e285cc60f31290c81f82ea13429c4f` |
| 41164 | fabert | 1 | Public | `class` | classification | `3afd98b3ff70400e7096ba55257947127abca4d3fb9f11bb9e465aafbc5329ed` |
| 41143 | jasmine | 1 | Public | `class` | classification | `08142863cfdf0754752dd67bfb3ee7d9e29030c9b7fca6c85d69927000858ae2` |
| 1169 | airlines | 1 | Public | `Delay` | classification | `c7683a90b40c2cabb1ddc950413da8efa231c058327b8c95511c40af13516c14` |
| 41158 | gina | 1 | Public | `class` | classification | `2511d829e62d473e1a9ee833770bfac65bdf9df321268a77e15db680afcd6a49` |
| 41144 | madeline | 1 | Public | `class` | classification | `8562d10a1b945172055dcd11852525962f952587eeee173790779c90af4f5ae4` |
| 41145 | philippine | 1 | Public | `class` | classification | `819470adb79ea428c9f7c1cd9e58ceb8ab77c4d2a3249f22393318c4d0d0484d` |
| 41156 | ada | 1 | Public | `class` | classification | `9f487bfee74bcf72538fcdc1a1336931e4437824843eef090fdf71506602a290` |
| 41157 | arcene | 2 | Public | `class` | classification | `c58c70465382108be4261cd6828b01c54e49fa53282e88d1417928c5f29ad66f` |
| 41168 | jannis | 1 | Public | `class` | classification | `ffc0530b7f98207096b603b33c5cb33d1f92bc8b4f2a62acbeb5fe70e4107d4c` |
| 1515 | micro-mass | 2 | Public | `Class` | classification | `b4109c3ac28c6d248993fb1936df6137414c9d90efb46179880948e4cb46aa9f` |
| 41146 | sylvine | 1 | Public | `class` | classification | `cbd39adc5849eb951434e7194090e2611ff8a34033dbb0a78dd7c4249791cf48` |
| 41142 | christine | 1 | Public | `class` | classification | `307099164edf07b914eeb2b968b0ef8e0393dedb7bfe0c04937aea49d9f949cd` |
| 41169 | helena | 1 | Public | `class` | classification | `351c371c164b5ceacd9a87697d9efde170daf51ce4b3ce3819f25b7912e1042a` |
| 41166 | volkert | 1 | Public | `class` | classification | `0b782b59f04b71397da65a05d1117095944d35450ec4d54c423eda6809122d29` |
| 40685 | shuttle | 1 | public | `class` | classification | `d0e47eb009fe0b77e67ec16a21d64efb69073dfb5f405936b7a51fc1dc22c9fd` |
| 41150 | MiniBooNE | 1 | CC0 | `signal` | classification | `907fae4b0de2f61e88f6cda799f53a9f7e39ee6876aa971a9211f5802d89a79d` |
| 41162 | kick | 1 | CC0 | `IsBadBuy` | classification | `918b922249d410f9274557850e0f494d51a450a97c4755ce0c6889b82811d0ff` |
| 42733 | Click_prediction_small | 10 | Public | `click` | classification | `add9a17b955be5ee4a89c3dc2c52544eb1f05098b404c99183d209caca4dbc39` |
| 42734 | okcupid-stem | 3 | Public | `job` | classification | `1af41ae7bd750443ea535b324b91e830799c13b9bc2c2f64ea4fff29634c024d` |
| 42732 | sf-police-incidents | 6 | CC0 | `ViolentCrime` | classification | `f6415889d9952c71226edceaba74f4f105416eb77977b632307d4a787e50d9bb` |
| 42570 | Mercedes_Benz_Greener_Manufacturing | 2 | Public | `y` | regression | `44cb0065e346847de80f8a2a0fdfa9f38cfff90275129f3419cffb420d690910` |
| 550 | quake | 2 | Public | `col_4` | regression | `73b599c906cc2b6ff39c99513679db5f6a9d3802b4a5da1dfabac4386b77ee92` |
| 546 | sensory | 1 | Public | `Score` | regression | `158968b205cce12fa1b58da519aece8a1e396a633ce5fbf6a960be4413b93d56` |
| 505 | tecator | 1 | Public | `fat` | regression | `47044ff39e813ee22441f718f33cca130686ac130bbf79756f8f7c5e9c2c05f5` |
| 41540 | black_friday | 1 | GPL-1 | `Purchase` | regression | `784e88ab33be4b7aae60a85ec8e1cad53ab3d27ef3350ada00fc93038d06b751` |
| 422 | topo_2_1 | 1 | Public | `oz267` | regression | `79021c5bf78310c1c941db24944559eb11526607354f605b98f31bdea2711617` |
| 416 | yprop_4_1 | 1 | Public | `oz252` | regression | `13bf426f16a95f257f345800152806178c1cfc25e5e1a399c8a4973519cf497b` |
| 42724 | OnlineNewsPopularity | 2 | Public | `shares` | regression | `bb6f42580b1a79c755c8d7652e13b8033b881d09c1c82ac8d721e5007b9cd84a` |
| 42729 | nyc-taxi-green-dec-2016 | 3 | Public | `tip_amount` | regression | `d5a584ef08e6dc02bf46cf20d2021f93fd687716b984e5a1b86a66d339814ba6` |
| 201 | pol | 1 | Public | `foo` | regression | `0f5f397fb144dfb7269a03e4eccc84a413a48982ceef0a91b142c9917b39675e` |
| 41980 | SAT11-HAND-runtime-regression | 1 | Public | `runtime` | regression | `1ab43811428dfbd685e41edd34fd63222d4da768ab5060c8ee820268740a848b` |
| 531 | boston | 1 | Public | `MEDV` | regression | `2826ddb8cbce925e98114491f4774544b17d70ff0b7b1eede2af0a2ffd85a4f0` |
| 42563 | house_prices_nominal | 1 | Public | `SalePrice` | regression | `1e83c38ec2f33615fc3308a641e6f3a1fe4a7b4fbb8dd54913c6082799602bdc` |
| 574 | house_16H | 1 | Public | `price` | regression | `c9d4c46cb5f598fd9e3e1993a1ff06d85c662e50a9fe1fef9c1dc7d067d3b311` |
| 3277 | QSAR-TID-10980 | 1 | Public | `MEDIAN_PXC50` | regression | `54d22f442cb7cc1cdd54ea60d549d68faab9da1e78bea3784b68ec47056252a7` |
| 43071 | MIP-2016-regression | 3 | Public | `PAR10` | regression | `8b80c61c8c093f4c336b92552afb0d51c6ccdd7b49bb75b763e992391da297a1` |
| 42 | soybean | 1 | Public | `class` | classification | `673e0965c2cc18e70b177866afc8fcd15ad553a0b2098229abb2e0b3d4312f33` |
| 60 | waveform-5000 | 1 | Public | `class` | classification | `ea857b7e905331181e43f56af95b3d512d08d0b03a65d2af48cb4dbdda2334a3` |
| 312 | scene | 1 | Public | `Urban` | classification | `bf94fc39f84c51b04cc0001f44e332e171c03d371faaa4611f579539d5b7de93` |
| 375 | JapaneseVowels | 1 | Public | `speaker` | classification | `2290186e0b8723a88c8e52e48c396f06f6cf98733f13d0e62d08b2b5c6184190` |
| 377 | synthetic_control | 1 | Public | `class` | classification | `2692c0c1bbf7b14485c94b151da9577f4b33acdd3352b448804f74f23e1dab39` |
| 451 | irish | 1 | Public | `Leaving_Certificate` | classification | `4d79ad272bdbdc5e4cdbcaa30e153aa50439d0e145a86c5ca9900d1224c502d2` |
| 470 | profb | 1 | Public | `Home/Away` | classification | `8a699382dcbf580a3b875644008d82171a2e54ef005bf20599b854cfb78cd23d` |
