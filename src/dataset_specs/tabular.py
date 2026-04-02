"""Declarative tabular dataset specs shared across the repo."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property


@dataclass(frozen=True)
class OHEBlockSpec:
    """One-hot block boundaries in flattened feature vectors."""

    start: int
    end: int


@dataclass(frozen=True)
class TabularDatasetSpec:
    """Schema-level metadata for a tabular dataset."""

    name: str
    feature_names: tuple[str, ...]
    target_name: str
    numerical_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    cardinalities: tuple[int, ...]
    immutable_features: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        feature_set = set(self.feature_names)
        numerical_set = set(self.numerical_features)
        categorical_set = set(self.categorical_features)

        if len(self.feature_names) != len(feature_set):
            raise ValueError(f"{self.name}: feature_names must be unique.")
        if numerical_set & categorical_set:
            raise ValueError(f"{self.name}: numerical_features and categorical_features must be disjoint.")
        if numerical_set | categorical_set != feature_set:
            raise ValueError(
                f"{self.name}: feature_names must equal numerical_features U categorical_features."
            )
        if len(self.cardinalities) != len(self.categorical_features):
            raise ValueError(
                f"{self.name}: cardinalities length must match categorical_features length."
            )
        if any(card <= 0 for card in self.cardinalities):
            raise ValueError(f"{self.name}: cardinalities must be positive integers.")
        if not set(self.immutable_features).issubset(feature_set):
            raise ValueError(f"{self.name}: immutable_features must be a subset of feature_names.")

    @cached_property
    def input_types(self) -> tuple[str, ...]:
        """Feature types in the original training feature order."""
        numerical_set = set(self.numerical_features)
        return tuple("numerical" if feature in numerical_set else "categorical" for feature in self.feature_names)

    @cached_property
    def feature_slices(self) -> tuple[tuple[int, int], ...]:
        """Encoded feature-space slice for each original feature."""
        slices: list[tuple[int, int]] = []
        position = 0
        cardinality_iter = iter(self.cardinalities)

        for feature_type in self.input_types:
            width = 1 if feature_type == "numerical" else next(cardinality_iter)
            slices.append((position, position + width))
            position += width

        return tuple(slices)

    @cached_property
    def categorical_slices(self) -> tuple[tuple[int, int], ...]:
        """Encoded slices for categorical one-hot blocks only."""
        return tuple(
            feature_slice
            for feature_type, feature_slice in zip(self.input_types, self.feature_slices)
            if feature_type == "categorical"
        )

    @cached_property
    def ohe_blocks(self) -> tuple[OHEBlockSpec, ...]:
        """Canonical OHE block specs for benchmark-time snapping."""
        return tuple(OHEBlockSpec(start=start, end=end) for start, end in self.categorical_slices)

    @cached_property
    def ohe_feature_types(self) -> tuple[str, ...]:
        """Feature types in encoded OHE space."""
        feature_types: list[str] = []
        cardinality_iter = iter(self.cardinalities)

        for feature_type in self.input_types:
            if feature_type == "numerical":
                feature_types.append("numerical")
            else:
                feature_types.extend(["categorical"] * next(cardinality_iter))

        return tuple(feature_types)

    @cached_property
    def n_features(self) -> int:
        """Encoded feature dimensionality after OHE expansion."""
        if not self.feature_slices:
            return 0
        return self.feature_slices[-1][1]


TABULAR_DATASET_SPECS: dict[str, TabularDatasetSpec] = {
    "adult": TabularDatasetSpec(
        name="adult",
        feature_names=(
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
        ),
        target_name="class",
        numerical_features=(
            "age",
            "fnlwgt",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ),
        categorical_features=(
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ),
        cardinalities=(7, 16, 7, 14, 6, 5, 2, 41),
    ),
    "compas": TabularDatasetSpec(
        name="compas",
        feature_names=(
            "age",
            "priors_count",
            "c_days_from_compas",
            "days_b_screening_arrest",
            "sex",
            "race",
            "c_charge_degree",
        ),
        target_name="two_year_recid",
        numerical_features=(
            "age",
            "priors_count",
            "c_days_from_compas",
            "days_b_screening_arrest",
        ),
        categorical_features=("sex", "race", "c_charge_degree"),
        cardinalities=(2, 6, 2),
    ),
    "german_credit": TabularDatasetSpec(
        name="german_credit",
        feature_names=(
            "duration",
            "credit_amount",
            "installment_rate",
            "present_residence",
            "age",
            "num_credits",
            "num_liable",
            "status",
            "credit_history",
            "purpose",
            "savings",
            "employment",
            "personal_status",
            "other_debtors",
            "property",
            "other_installment_plans",
            "housing",
            "job",
            "telephone",
            "foreign_worker",
        ),
        target_name="target",
        numerical_features=(
            "duration",
            "credit_amount",
            "installment_rate",
            "present_residence",
            "age",
            "num_credits",
            "num_liable",
        ),
        categorical_features=(
            "status",
            "credit_history",
            "purpose",
            "savings",
            "employment",
            "personal_status",
            "other_debtors",
            "property",
            "other_installment_plans",
            "housing",
            "job",
            "telephone",
            "foreign_worker",
        ),
        cardinalities=(4, 5, 10, 5, 5, 4, 3, 4, 3, 3, 4, 2, 2),
    ),
    "lending_club": TabularDatasetSpec(
        name="lending_club",
        feature_names=(
            "loan_amnt",
            "int_rate",
            "annual_inc",
            "dti",
            "delinq_2yrs",
            "open_acc",
            "pub_rec",
            "revol_util",
            "term",
            "grade",
            "home_ownership",
            "verification_status",
        ),
        target_name="loan_status",
        numerical_features=(
            "loan_amnt",
            "int_rate",
            "annual_inc",
            "dti",
            "delinq_2yrs",
            "open_acc",
            "pub_rec",
            "revol_util",
        ),
        categorical_features=("term", "grade", "home_ownership", "verification_status"),
        cardinalities=(2, 7, 5, 3),
    ),
    "heloc": TabularDatasetSpec(
        name="heloc",
        feature_names=(
            "estimate_of_risk",
            "months_since_first_trade",
            "months_since_last_trade",
            "average_duration_of_resolution",
            "number_of_satisfactory_trades",
            "nr_trades_insolvent_for_over_60_days",
            "nr_trades_insolvent_for_over_90_days",
            "percentage_of_legal_trades",
            "months_since_last_illegal_trade",
            "maximum_illegal_trades_over_last_year",
            "maximum_illegal_trades",
            "nr_total_trades",
            "nr_trades_initiated_in_last_year",
            "percentage_of_installment_trades",
            "months_since_last_inquiry_not_recent",
            "nr_inquiries_in_last_6_months",
            "nr_inquiries_in_last_6_months_not_recent",
            "net_fraction_of_revolving_burden",
            "net_fraction_of_installment_burden",
            "nr_revolving_trades_with_balance",
            "nr_installment_trades_with_balance",
            "nr_banks_with_high_ratio",
            "percentage_trades_with_balance",
        ),
        target_name="is_at_risk",
        numerical_features=(
            "estimate_of_risk",
            "months_since_first_trade",
            "months_since_last_trade",
            "average_duration_of_resolution",
            "number_of_satisfactory_trades",
            "nr_trades_insolvent_for_over_60_days",
            "nr_trades_insolvent_for_over_90_days",
            "percentage_of_legal_trades",
            "months_since_last_illegal_trade",
            "maximum_illegal_trades_over_last_year",
            "maximum_illegal_trades",
            "nr_total_trades",
            "nr_trades_initiated_in_last_year",
            "percentage_of_installment_trades",
            "months_since_last_inquiry_not_recent",
            "nr_inquiries_in_last_6_months",
            "nr_inquiries_in_last_6_months_not_recent",
            "net_fraction_of_revolving_burden",
            "net_fraction_of_installment_burden",
            "nr_revolving_trades_with_balance",
            "nr_installment_trades_with_balance",
            "nr_banks_with_high_ratio",
            "percentage_trades_with_balance",
        ),
        categorical_features=(),
        cardinalities=(),
    ),
    "give_me_some_credit": TabularDatasetSpec(
        name="give_me_some_credit",
        feature_names=(
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfDependents",
        ),
        target_name="SeriousDlqin2yrs",
        numerical_features=(
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfDependents",
        ),
        categorical_features=(),
        cardinalities=(),
    ),
}


def get_tabular_dataset_spec(name: str) -> TabularDatasetSpec:
    """Return the registered tabular dataset spec."""
    try:
        return TABULAR_DATASET_SPECS[name]
    except KeyError as exc:
        available = ", ".join(sorted(TABULAR_DATASET_SPECS))
        raise KeyError(f"Unknown tabular dataset spec '{name}'. Available: [{available}]") from exc
