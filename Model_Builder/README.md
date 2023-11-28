# Model Builder Samples
This repository contains Decision Optimization samples that you can use in the Model Builder user interface in:
- IBM watsonx.ai and IBM Cloud Pak for Data as a Service
- IBM Cloud Pak for Data V4.8.x, V4.7.x, V4.6.x, V4.5.x and V4.0.x
- IBM Cloud Pak for Data (previous unsupported versions) V3.5.x, V3.0.x and V2.5.x

For details on **how to use these samples**, refer to the *Examples and Samples* section of the Decision Optimization Building models documentation for your product.



## Repository contents
The `watsonx.ai and Cloud Pak for Data as a Service` directory contains `.zip` files for the samples that can be used with IBM watsonx.ai and IBM Cloud Pak for Data as a Service.

The `Cloud Pak for Data v4.8.x` directory contains `.zip` files for the samples that can be used with IBM Cloud Pak for Data V4.8.x.

The `Cloud Pak for Data v4.7.x` directory contains `.zip` files for the samples that can be used with IBM Cloud Pak for Data V4.7.x.

The `Cloud Pak for Data v4.6.x` directory contains `.zip` files for the samples that can be used with IBM Cloud Pak for Data V4.6.x.

The `Cloud Pak for Data v4.5.x` directory contains `.zip` files for the samples that can be used with IBM Cloud Pak for Data V4.5.x.

The `Cloud Pak for Data v4.0.x` directory contains `.zip` files for the samples that can be used with IBM Cloud Pak for Data V4.0.x.
(TalentCPO and DietLP samples can only be used with Cloud Pak for Data v4.0.3 or later.)

The `Previous versions - no longer supported` directory contains previous versions of IBM Cloud Pak for Data (V3.5.x, V3.0.x and V2.5.x).

Each sample zip file contains one or more **scenarios** which provide the Decision Optimization model (in Python, OPL or using the Modeling Assistant) as well as the data in `.csv` files. Some samples also contain visualization files.



## Decision Optimization models
The following table lists the Decision Optimization samples that are provided in these directories. All these assets use the Decision Optimization experiment UI and contain data.

| Name | Problem type | Model Type |
|------|--------------|------------|
| BridgeScheduling | Scheduling | Modeling Assistant |
| Diet | Blending | Python |
| DietLP | Blending | LP (CPLEX) |
| EnvironmentAndExtension | Using an environment with an extension that contains a library file and YAML code. | Python |
| HouseConstructionScheduling | Scheduling with assignment | Modeling Assistant |
| IntermediateSolutions | Enabling intermediate solutions for CPLEX and CPO models  | Python |
| MarketingCampaignAssignment | Resource Assignment (Scenarios 1 - 4) | Modeling Assistant |
| MarketingCampaignAssignment | Selection and Allocation (Scenario 4 - Selection) | Modeling Assistant |
| Multifiles | Using a model with multiple files | Python and LP |
| PastaProduction | Production | OPL |
| PortfolioAllocation | Selection & Allocation | Modeling Assistant |
| PythonEngineSettings | Geometrical problem with customized engine settings | Python |
| ShiftAssignment | Resource Assignment with custom decisions and a custom constraint | Modeling Assistant |
| StaffPlanning | Multi-Scenario Planning (to be used with CopyAndSolveScenarios.ipynb) | Python |
| SupplyDemandPlanning | Supply & Demand Planning | Modeling Assistant |
| TalentCPO | Movie scheduling | CPO (CP Optimizer) |



## To use these samples in Watson Studio
1. Download and extract all the DO-samples on to your computer. You can also download just the one sample, but in this case do not extract it.
2. **Create a project** in IBM Watson Studio. Select Create an empty project, enter a project name and click Create.
3. (*For Cloud Pak for Data as a Service only*): In the Overview tab of your project, click add a Machine Learning service and select an existing service instance (or create a new one) and click Select.
4. Click **Add to Project**.
5. Select **Decision Optimization experiment**.
6. Select the **From file** tab in the Decision Optimization experiment pane that opens.
7. Click **Add file**. Then browse to the Model_Builder folder in your downloaded DO-samples. Select the relevant product and version subfolder. Choose your sample .zip file.
8. Choose a **deployment space** from the drop-down menu (or create one) and click **Create**.
9. (*For Cloud Pak for Data as a Service only*): If you haven't already associated a Machine Learning service with your project, you must first select Add a service to select or create one, before choosing your deployment space for your experiment.
10. Click **Create**.
