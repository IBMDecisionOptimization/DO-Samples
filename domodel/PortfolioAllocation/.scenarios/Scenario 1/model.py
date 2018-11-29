from docplex.mp.model import *
from docplex.mp.utils import *
from docloud.status import JobSolveStatus
from docplex.mp.conflict_refiner import ConflictRefiner, VarUbConstraintWrapper, VarLbConstraintWrapper
import time
import sys
import operator

import pandas as pd
import numpy as np


# Label constraint
def helper_add_labeled_cplex_constraint(mdl, expr, label, context=None, columns=None):
    global expr_counter
    if isinstance(expr, bool):
        pass  # Adding a trivial constraint: if infeasible, docplex will raise an exception it is added to the model
    else:
        expr.name = '_L_EXPR_' + str(len(expr_to_info) + 1)
        if columns:
            ctxt = ", ".join(str(getattr(context, col)) for col in columns)
        else:
            if context:
                ctxt = context.Index if isinstance(context.Index, str) is not None else ", ".join(context.Index)
            else:
                ctxt = None
        expr_to_info[expr.name] = (label, ctxt)
    mdl.add(expr)

def helper_get_index_names_for_type(dataframe, type):
    if not is_pandas_dataframe(dataframe):
        return None
    return [name for name in dataframe.index.names if name in helper_concept_id_to_index_names_map.get(type, [])]


helper_concept_id_to_index_names_map = {
    'cItem': ['id_of_Investment'],
    'country': ['id_of_Country'],
    'param': ['id_of_Param'],
    'investment': ['id_of_Investment'],
    'recommendation': ['id_of_Recommendation'],
    'industry': ['id_of_Industry']}


# Data model definition for each table
# Data collection: list_of_Country ['id']
# Data collection: list_of_Industry ['id']
# Data collection: list_of_Investment ['id', 'country', 'expected_return', 'industry', 'recommendation', 'stock_price']
# Data collection: list_of_Param ['budget', 'wealth_prct']
# Data collection: list_of_Recommendation ['id']

# Create a pandas Dataframe for each data table
# --- Handling table for implicit concept
list_of_Country = pd.DataFrame(inputs['investment']['country'].unique(), columns=['id']).dropna()
# --- Handling table for implicit concept
list_of_Industry = pd.DataFrame(inputs['investment']['industry'].unique(), columns=['id']).dropna()
list_of_Investment = inputs['investment']
list_of_Investment = list_of_Investment[['id', 'country', 'expected return', 'industry', 'recommendation', 'stock price']].copy()
list_of_Investment.rename(columns={'id': 'id', 'country': 'country', 'expected return': 'expected_return', 'industry': 'industry', 'recommendation': 'recommendation', 'stock price': 'stock_price'}, inplace=True)
list_of_Param = inputs['param']
list_of_Param = list_of_Param[['budget', 'wealth prct']].copy()
list_of_Param.rename(columns={'budget': 'budget', 'wealth prct': 'wealth_prct'}, inplace=True)
# --- Handling table for implicit concept
list_of_Recommendation = pd.DataFrame(inputs['investment']['recommendation'].unique(), columns=['id']).dropna()

# Set index when a primary key is defined
list_of_Country.set_index('id', inplace=True)
list_of_Country.sort_index(inplace=True)
list_of_Country.index.name = 'id_of_Country'
list_of_Industry.set_index('id', inplace=True)
list_of_Industry.sort_index(inplace=True)
list_of_Industry.index.name = 'id_of_Industry'
list_of_Investment.set_index('id', inplace=True)
list_of_Investment.sort_index(inplace=True)
list_of_Investment.index.name = 'id_of_Investment'
list_of_Param.set_index('wealth_prct', inplace=True)
list_of_Param.sort_index(inplace=True)
list_of_Param.index.name = 'id_of_Param'
list_of_Recommendation.set_index('id', inplace=True)
list_of_Recommendation.sort_index(inplace=True)
list_of_Recommendation.index.name = 'id_of_Recommendation'






def build_model():
    mdl = Model()

    # Definition of model variables
    list_of_Investment['allocationVar'] = mdl.continuous_var_list(len(list_of_Investment))
    list_of_Investment['selectionVar'] = mdl.binary_var_list(len(list_of_Investment))
    list_of_Investment['integerSubAllocationVar'] = mdl.integer_var_list(len(list_of_Investment))


    # Definition of model
    # Objective (Guided) Maximize total expected return of investment of allocation-
    # Combine weighted criteria: 
    # 	cMaximizeGoalSelect cMaximizeGoalSelect{
    # 	cSingleCriterionGoal.goalFilter = null,
    # 	cSingleCriterionGoal.numericExpr = total cAllocation[investment] / investment / expected return,
    # 	cScaledGoal.scaleFactorExpr = 1} with weight 5.0
    # 	cMinimizeCovariance cMinimizeCovariance{
    # 	cSingleCriterionGoal.goalFilter = null,
    # 	cMinimizeCovariance.multiplier = 1,
    # 	cSingleCriterionGoal.numericExpr = total 1 * total covariance / value * count( covariance / investment_1 / inverse(cAllocation[investment].investment)) * count( covariance / investment_2 / inverse(cAllocation[investment].investment)) over covariance,
    # 	cMinimizeCovariance.allocation = cAllocation[investment],
    # 	cMinimizeCovariance.covarianceMatrix = covariance,
    # 	cScaledGoal.scaleFactorExpr = param / rho / 2.0} with weight 5.0
    list_of_Investment['conditioned_expected_return'] = list_of_Investment.allocationVar * list_of_Investment.expected_return
    agg_Investment_conditioned_expected_return_SG1 = mdl.sum(list_of_Investment.conditioned_expected_return)
    
    kpis_expression_list = [
        (1, 1.0, agg_Investment_conditioned_expected_return_SG1, 1, 0, 'total expected return of allocated investments over all allocations')]
    custom_code.update_goals_list(kpis_expression_list)
    
    for _, kpi_weight, kpi_expr, kpi_factor, kpi_offset, kpi_name in kpis_expression_list:
        mdl.add_kpi(kpi_weight * ((kpi_expr * kpi_factor) - kpi_offset), publish_name=kpi_name)
    
    mdl.maximize(sum([kpi_sign * kpi_weight * ((kpi_expr * kpi_factor) - kpi_offset) for kpi_sign, kpi_weight, kpi_expr, kpi_factor, kpi_offset, kpi_name in kpis_expression_list]))
    
    # [ST_1] Constraint : (Guided) Synchronize selection with allocation_cIterativeRelationalConstraint
    # Synchronize selection with investment allocations
    # Label: CT_1_Synchronize_selection_with_investment_allocations
    join_Investment = list_of_Investment.reset_index().merge(list_of_Investment.reset_index(), left_on=['id_of_Investment'], right_on=['id_of_Investment'], suffixes=('', '_right')).set_index(['id_of_Investment'])
    groupbyLevels = [join_Investment.index.names.index(name) for name in list_of_Investment.index.names]
    groupby_Investment = join_Investment.allocationVar.groupby(level=groupbyLevels[0]).sum().to_frame(name='allocationVar')
    list_of_Investment_maxValueAllocation = pd.Series([1000] * len(list_of_Investment)).to_frame('maxValueAllocation').set_index(list_of_Investment.index)
    join_Investment_2 = list_of_Investment.join(list_of_Investment_maxValueAllocation.maxValueAllocation, how='inner')
    join_Investment_2['conditioned_maxValueAllocation'] = join_Investment_2.selectionVar * join_Investment_2.maxValueAllocation
    join_Investment_3 = groupby_Investment.join(join_Investment_2.conditioned_maxValueAllocation, how='inner')
    for row in join_Investment_3.itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.allocationVar <= row.conditioned_maxValueAllocation, 'Synchronize selection with investment allocations', row)
    
    # [ST_2] Constraint : (Guided) Synchronize selection with allocation_cIterativeRelationalConstraint
    # Synchronize selection with investment allocations
    # Label: CT_2_Synchronize_selection_with_investment_allocations
    join_Investment = list_of_Investment.reset_index().merge(list_of_Investment.reset_index(), left_on=['id_of_Investment'], right_on=['id_of_Investment'], suffixes=('', '_right')).set_index(['id_of_Investment'])
    groupbyLevels = [join_Investment.index.names.index(name) for name in list_of_Investment.index.names]
    groupby_Investment = join_Investment.allocationVar.groupby(level=groupbyLevels[0]).sum().to_frame(name='allocationVar')
    list_of_Investment_minValueAllocationForAssignment = pd.Series([10] * len(list_of_Investment)).to_frame('minValueAllocationForAssignment').set_index(list_of_Investment.index)
    join_Investment_2 = list_of_Investment.join(list_of_Investment_minValueAllocationForAssignment.minValueAllocationForAssignment, how='inner')
    join_Investment_2['conditioned_minValueAllocationForAssignment'] = join_Investment_2.selectionVar * join_Investment_2.minValueAllocationForAssignment
    join_Investment_3 = groupby_Investment.join(join_Investment_2.conditioned_minValueAllocationForAssignment, how='inner')
    for row in join_Investment_3.itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.allocationVar >= row.conditioned_minValueAllocationForAssignment, 'Synchronize selection with investment allocations', row)
    
    # [ST_3] Constraint : (Guided) All allocation must be an integer number of Stock price of investment_cIterativeRelationalConstraint
    # All investment allocations must be an integer number of stock price
    # Label: CT_3_All_investment_allocations_must_be_an_integer_number_of_stock_price
    bin_op_Investment = pd.Series(list_of_Investment.integerSubAllocationVar * list_of_Investment.stock_price, name='result').to_frame()[list_of_Investment.stock_price.notnull()]
    join_Investment = list_of_Investment.join(bin_op_Investment.result, how='inner')
    for row in join_Investment.itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.allocationVar == row.result, 'All investment allocations must be an integer number of stock price', row)
    
    # [ST_4] Constraint : (Guided) ensure that total allocation of investment <= budget of param_cGlobalRelationalConstraint
    # total allocation over all investments is less than or equal to budget of param
    # Label: CT_4_total_allocation_over_all_investments_is_less_than_or_equal_to_budget_of_param
    agg_Investment_allocationVar_lhs = mdl.sum(list_of_Investment.allocationVar)
    agg_Param_budget_rhs = sum(list_of_Param.budget)
    helper_add_labeled_cplex_constraint(mdl, agg_Investment_allocationVar_lhs <= agg_Param_budget_rhs, 'total allocation over all investments is less than or equal to budget of param')
    
    # [ST_5] Constraint : (Guided) No selection of investment where recommendation of investment is "Strong Sell"_cIterativeRelationalConstraint
    # No selection of investments where recommendation is "Strong Sell"
    # Label: CT_5_No_selection_of_investments_where_recommendation_is__Strong_Sell_
    filtered_Investment = list_of_Investment[list_of_Investment.recommendation == 'Strong Sell'].copy()
    for row in filtered_Investment.itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.selectionVar <= 0, 'No selection of investments where recommendation is "Strong Sell"', row)
    
    # [ST_6] Constraint : (Guided) for each Country, number of selection of investments of country <= 3_cIterativeRelationalConstraint
    # For each country, number of selections of investments is less than or equal to 3
    # Label: CT_6_For_each_country__number_of_selections_of_investments_is_less_than_or_equal_to_3
    join_Country = list_of_Country.reset_index().merge(list_of_Investment.reset_index(), left_on=['id_of_Country'], right_on=['country']).set_index(['id_of_Country', 'id_of_Investment'])
    groupbyLevels = [join_Country.index.names.index(name) for name in list_of_Country.index.names]
    groupby_Country = join_Country.selectionVar.groupby(level=groupbyLevels).sum().to_frame()
    for row in groupby_Country.itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.selectionVar <= 3, 'For each country, number of selections of investments is less than or equal to 3', row)
    
    # [ST_7] Constraint : (Guided) for each Industry, number of selection of investments of Industry <= 2_cIterativeRelationalConstraint
    # For each industry, number of selections of investments is less than or equal to 2
    # Label: CT_7_For_each_industry__number_of_selections_of_investments_is_less_than_or_equal_to_2
    join_Industry = list_of_Industry.reset_index().merge(list_of_Investment.reset_index(), left_on=['id_of_Industry'], right_on=['industry']).set_index(['id_of_Industry', 'id_of_Investment'])
    groupbyLevels = [join_Industry.index.names.index(name) for name in list_of_Industry.index.names]
    groupby_Industry = join_Industry.selectionVar.groupby(level=groupbyLevels).sum().to_frame()
    for row in groupby_Industry.itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.selectionVar <= 2, 'For each industry, number of selections of investments is less than or equal to 2', row)
    
    # [ST_8] Constraint : (Guided) for each Country, total allocation of investments of country <= 500_cIterativeRelationalConstraint
    # For each country, total allocation of investments is less than or equal to 500
    # Label: CT_8_For_each_country__total_allocation_of_investments_is_less_than_or_equal_to_500
    join_Country = list_of_Country.reset_index().merge(list_of_Investment.reset_index(), left_on=['id_of_Country'], right_on=['country']).set_index(['id_of_Country', 'id_of_Investment'])
    groupbyLevels = [join_Country.index.names.index(name) for name in list_of_Country.index.names]
    groupby_Country = join_Country.allocationVar.groupby(level=groupbyLevels).sum().to_frame()
    for row in groupby_Country.itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.allocationVar <= 500, 'For each country, total allocation of investments is less than or equal to 500', row)
    
    # [ST_9] Constraint : (Guided) for each investment, allocation of investment <= 300_cIterativeRelationalConstraint
    # For each investment, allocation is less than or equal to 300
    # Label: CT_9_For_each_investment__allocation_is_less_than_or_equal_to_300
    for row in list_of_Investment.itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.allocationVar <= 300, 'For each investment, allocation is less than or equal to 300', row)
    
    # [ST_10] Constraint : (Guided) ensure that total allocation of investments of country : FR <= 150_cGlobalRelationalConstraint
    # total allocation of investments of FR is less than or equal to 150
    # Label: CT_10_total_allocation_of_investments_of_FR_is_less_than_or_equal_to_150
    filtered_Country_lhs = list_of_Country.loc[['FR']]
    join_Country_lhs = filtered_Country_lhs.reset_index().merge(list_of_Investment.reset_index(), left_on=['id_of_Country'], right_on=['country']).set_index(['id_of_Country', 'id_of_Investment'])
    agg_Country_allocationVar_lhs = mdl.sum(join_Country_lhs.allocationVar)
    helper_add_labeled_cplex_constraint(mdl, agg_Country_allocationVar_lhs <= 150, 'total allocation of investments of FR is less than or equal to 150')
    
    # [ST_11] Constraint : (Guided) ensure that number of selection <= 6_cGlobalRelationalConstraint
    # number of investment selections is less than or equal to 6
    # Label: CT_11_number_of_investment_selections_is_less_than_or_equal_to_6
    agg_Investment_selectionVar_lhs = mdl.sum(list_of_Investment.selectionVar)
    helper_add_labeled_cplex_constraint(mdl, agg_Investment_selectionVar_lhs <= 6, 'number of investment selections is less than or equal to 6')


    return mdl


def solve_model(mdl):
    mdl.parameters.timelimit = 120
    # Call to custom code to update parameters value
    custom_code.update_solver_params(mdl.parameters)
    # Update parameters value based on environment variables definition
    cplex_param_env_prefix = 'ma.cplex.'
    cplex_params = [name.qualified_name for name in mdl.parameters.generate_params()]
    for param in cplex_params:
        env_param = cplex_param_env_prefix + param
        param_value = get_environment().get_parameter(env_param)
        if param_value:
            # Updating parameter value
            print("Updated value for parameter %s = %s" % (param, param_value))
            parameters = mdl.parameters
            for p in param.split('.')[1:]:
                parameters = parameters.__getattribute__(p)
            parameters.set(param_value)

    msol = mdl.solve(log_output=True)
    if not msol:
        print("!!! Solve of the model fails")
        if mdl.get_solve_status() == JobSolveStatus.INFEASIBLE_SOLUTION or mdl.get_solve_status() == JobSolveStatus.INFEASIBLE_OR_UNBOUNDED_SOLUTION:
            crefiner = ConflictRefiner()
            conflicts = crefiner.refine_conflict(model, log_output=True)
            export_conflicts(conflicts)
    print 'Solve status: ', mdl.get_solve_status()
    mdl.report()
    return msol


expr_to_info = {}


def export_conflicts(conflicts):
    # Display conflicts in console
    print('Conflict set:')
    list_of_conflicts = pd.DataFrame(columns=['constraint', 'context', 'detail'])
    for conflict, index in zip(conflicts, range(len(conflicts))):
        st = conflict.status
        ct = conflict.element
        label, context = expr_to_info.get(conflict.name, ('N/A', conflict.name))
        label_type = type(conflict.element)
        if isinstance(conflict.element, VarLbConstraintWrapper) \
                or isinstance(conflict.element, VarUbConstraintWrapper):
            ct = conflict.element.get_constraint()

        # Print conflict information in console
        print("Conflict involving constraint: %s, \tfor: %s -> %s" % (label, context, ct))
        list_of_conflicts = list_of_conflicts.append(pd.DataFrame({'constraint': label, 'context': str(context), 'detail': ct},
                                                                  index=[index], columns=['constraint', 'context', 'detail']))

    # Update of the ``outputs`` dict must take the 'Lock' to make this action atomic,
    # in case the job is aborted
    global output_lock
    with output_lock:
        outputs['list_of_conflicts'] = list_of_conflicts


def export_solution(msol):
    start_time = time.time()
    list_of_Investment_solution = pd.DataFrame(index=list_of_Investment.index)
    list_of_Investment_solution['allocationVar'] = msol.get_values(list_of_Investment.allocationVar.values)
    list_of_Investment_solution['selectionVar'] = msol.get_values(list_of_Investment.selectionVar.values)
    list_of_Investment_solution['integerSubAllocationVar'] = msol.get_values(list_of_Investment.integerSubAllocationVar.values)

    # Update of the ``outputs`` dict must take the 'Lock' to make this action atomic,
    # in case the job is aborted
    global output_lock
    with output_lock:
        outputs['list_of_Investment_solution'] = list_of_Investment_solution.reset_index()
        custom_code.post_process_solution(msol, outputs)

    elapsed_time = time.time() - start_time
    print('solution export done in ' + str(elapsed_time) + ' secs')
    return


# Import custom code definition if module exists
try:
    from custom_code import CustomCode
    custom_code = CustomCode(globals())
except ImportError:
    # Create a dummy anonymous object for custom_code
    custom_code = type('', (object,), {'preprocess': (lambda *args: None),
                                       'update_goals_list': (lambda *args: None),
                                       'update_model': (lambda *args: None),
                                       'update_solver_params': (lambda *args: None),
                                       'post_process_solution': (lambda *args: None)})()

# Custom pre-process
custom_code.preprocess()

print('* building wado model')
start_time = time.time()
model = build_model()

# Model customization
custom_code.update_model(model)

elapsed_time = time.time() - start_time
print('model building done in ' + str(elapsed_time) + ' secs')

print('* running wado model')
start_time = time.time()
msol = solve_model(model)
elapsed_time = time.time() - start_time
print('model solve done in ' + str(elapsed_time) + ' secs')
if msol:
    export_solution(msol)
