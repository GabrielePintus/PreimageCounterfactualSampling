# Constrained Benchmark Feature Names

Feature-name reference for deciding which variables to lock in constrained
benchmark runs.

Notes:
- Names are original dataset feature names from `dataset_specs`.
- Encoded dimensions are after one-hot expansion.
- Locking a categorical feature freezes the whole one-hot block.

## Adult

- Original features: `14`
- Encoded dimensionality: `104`

| Feature | Type | Encoded slice |
|---|---|---:|
| `age` | numerical | `[0, 1)` |
| `workclass` | categorical | `[1, 8)` |
| `fnlwgt` | numerical | `[8, 9)` |
| `education` | categorical | `[9, 25)` |
| `education-num` | numerical | `[25, 26)` |
| `marital-status` | categorical | `[26, 33)` |
| `occupation` | categorical | `[33, 47)` |
| `relationship` | categorical | `[47, 53)` |
| `race` | categorical | `[53, 58)` |
| `sex` | categorical | `[58, 60)` |
| `capital-gain` | numerical | `[60, 61)` |
| `capital-loss` | numerical | `[61, 62)` |
| `hours-per-week` | numerical | `[62, 63)` |
| `native-country` | categorical | `[63, 104)` |

## COMPAS

- Original features: `7`
- Encoded dimensionality: `14`

| Feature | Type | Encoded slice |
|---|---|---:|
| `age` | numerical | `[0, 1)` |
| `priors_count` | numerical | `[1, 2)` |
| `c_days_from_compas` | numerical | `[2, 3)` |
| `days_b_screening_arrest` | numerical | `[3, 4)` |
| `sex` | categorical | `[4, 6)` |
| `race` | categorical | `[6, 12)` |
| `c_charge_degree` | categorical | `[12, 14)` |

## German Credit

- Original features: `20`
- Encoded dimensionality: `61`

| Feature | Type | Encoded slice |
|---|---|---:|
| `duration` | numerical | `[0, 1)` |
| `credit_amount` | numerical | `[1, 2)` |
| `installment_rate` | numerical | `[2, 3)` |
| `present_residence` | numerical | `[3, 4)` |
| `age` | numerical | `[4, 5)` |
| `num_credits` | numerical | `[5, 6)` |
| `num_liable` | numerical | `[6, 7)` |
| `status` | categorical | `[7, 11)` |
| `credit_history` | categorical | `[11, 16)` |
| `purpose` | categorical | `[16, 26)` |
| `savings` | categorical | `[26, 31)` |
| `employment` | categorical | `[31, 36)` |
| `personal_status` | categorical | `[36, 40)` |
| `other_debtors` | categorical | `[40, 43)` |
| `property` | categorical | `[43, 47)` |
| `other_installment_plans` | categorical | `[47, 50)` |
| `housing` | categorical | `[50, 53)` |
| `job` | categorical | `[53, 57)` |
| `telephone` | categorical | `[57, 59)` |
| `foreign_worker` | categorical | `[59, 61)` |

## Give Me Some Credit

- Original features: `10`
- Encoded dimensionality: `10`

| Feature | Type | Encoded slice |
|---|---|---:|
| `RevolvingUtilizationOfUnsecuredLines` | numerical | `[0, 1)` |
| `age` | numerical | `[1, 2)` |
| `NumberOfTime30-59DaysPastDueNotWorse` | numerical | `[2, 3)` |
| `DebtRatio` | numerical | `[3, 4)` |
| `MonthlyIncome` | numerical | `[4, 5)` |
| `NumberOfOpenCreditLinesAndLoans` | numerical | `[5, 6)` |
| `NumberOfTimes90DaysLate` | numerical | `[6, 7)` |
| `NumberRealEstateLoansOrLines` | numerical | `[7, 8)` |
| `NumberOfTime60-89DaysPastDueNotWorse` | numerical | `[8, 9)` |
| `NumberOfDependents` | numerical | `[9, 10)` |

## HELOC

- Original features: `23`
- Encoded dimensionality: `23`

| Feature | Type | Encoded slice |
|---|---|---:|
| `estimate_of_risk` | numerical | `[0, 1)` |
| `months_since_first_trade` | numerical | `[1, 2)` |
| `months_since_last_trade` | numerical | `[2, 3)` |
| `average_duration_of_resolution` | numerical | `[3, 4)` |
| `number_of_satisfactory_trades` | numerical | `[4, 5)` |
| `nr_trades_insolvent_for_over_60_days` | numerical | `[5, 6)` |
| `nr_trades_insolvent_for_over_90_days` | numerical | `[6, 7)` |
| `percentage_of_legal_trades` | numerical | `[7, 8)` |
| `months_since_last_illegal_trade` | numerical | `[8, 9)` |
| `maximum_illegal_trades_over_last_year` | numerical | `[9, 10)` |
| `maximum_illegal_trades` | numerical | `[10, 11)` |
| `nr_total_trades` | numerical | `[11, 12)` |
| `nr_trades_initiated_in_last_year` | numerical | `[12, 13)` |
| `percentage_of_installment_trades` | numerical | `[13, 14)` |
| `months_since_last_inquiry_not_recent` | numerical | `[14, 15)` |
| `nr_inquiries_in_last_6_months` | numerical | `[15, 16)` |
| `nr_inquiries_in_last_6_months_not_recent` | numerical | `[16, 17)` |
| `net_fraction_of_revolving_burden` | numerical | `[17, 18)` |
| `net_fraction_of_installment_burden` | numerical | `[18, 19)` |
| `nr_revolving_trades_with_balance` | numerical | `[19, 20)` |
| `nr_installment_trades_with_balance` | numerical | `[20, 21)` |
| `nr_banks_with_high_ratio` | numerical | `[21, 22)` |
| `percentage_trades_with_balance` | numerical | `[22, 23)` |

## Lending Club

- Original features: `12`
- Encoded dimensionality: `25`

| Feature | Type | Encoded slice |
|---|---|---:|
| `loan_amnt` | numerical | `[0, 1)` |
| `int_rate` | numerical | `[1, 2)` |
| `annual_inc` | numerical | `[2, 3)` |
| `dti` | numerical | `[3, 4)` |
| `delinq_2yrs` | numerical | `[4, 5)` |
| `open_acc` | numerical | `[5, 6)` |
| `pub_rec` | numerical | `[6, 7)` |
| `revol_util` | numerical | `[7, 8)` |
| `term` | categorical | `[8, 10)` |
| `grade` | categorical | `[10, 17)` |
| `home_ownership` | categorical | `[17, 22)` |
| `verification_status` | categorical | `[22, 25)` |

## Wisconsin Breast Cancer

- Original features: `30`
- Encoded dimensionality: `30`

| Feature | Type | Encoded slice |
|---|---|---:|
| `mean_radius` | numerical | `[0, 1)` |
| `mean_texture` | numerical | `[1, 2)` |
| `mean_perimeter` | numerical | `[2, 3)` |
| `mean_area` | numerical | `[3, 4)` |
| `mean_smoothness` | numerical | `[4, 5)` |
| `mean_compactness` | numerical | `[5, 6)` |
| `mean_concavity` | numerical | `[6, 7)` |
| `mean_concave_points` | numerical | `[7, 8)` |
| `mean_symmetry` | numerical | `[8, 9)` |
| `mean_fractal_dimension` | numerical | `[9, 10)` |
| `radius_error` | numerical | `[10, 11)` |
| `texture_error` | numerical | `[11, 12)` |
| `perimeter_error` | numerical | `[12, 13)` |
| `area_error` | numerical | `[13, 14)` |
| `smoothness_error` | numerical | `[14, 15)` |
| `compactness_error` | numerical | `[15, 16)` |
| `concavity_error` | numerical | `[16, 17)` |
| `concave_points_error` | numerical | `[17, 18)` |
| `symmetry_error` | numerical | `[18, 19)` |
| `fractal_dimension_error` | numerical | `[19, 20)` |
| `worst_radius` | numerical | `[20, 21)` |
| `worst_texture` | numerical | `[21, 22)` |
| `worst_perimeter` | numerical | `[22, 23)` |
| `worst_area` | numerical | `[23, 24)` |
| `worst_smoothness` | numerical | `[24, 25)` |
| `worst_compactness` | numerical | `[25, 26)` |
| `worst_concavity` | numerical | `[26, 27)` |
| `worst_concave_points` | numerical | `[27, 28)` |
| `worst_symmetry` | numerical | `[28, 29)` |
| `worst_fractal_dimension` | numerical | `[29, 30)` |
