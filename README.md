# <img src="docs/assets/SynthCraft.png" height=25> SynthCraft

# 📦 Installation: SynthCraft
The installation process is as follows:

## 🐍 Set up the conda environments
SynthCraft uses conda to manage the Python environments. Before installing SynthCraft, you need to set up two conda environments as follows.

If you do not have conda (“Anaconda”) installed on your system, you should [install the miniconda distribution](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html).

Create the main conda environment for SynthCraft:
### Create the environment:
```
conda create -n SynthCraft python=3.9 -y
conda install conda-forge::weasyprint
```
### Create a separate conda environment that will be used for code execution:
```
conda create -n climb-code python=3.9 -y
```

### Activate the environment:
```
conda activate climb-code
```

### Install some standard packages in the environment. If more packages are needed by generated code, those will be automatically installed by the tool.
```
conda install pandas numpy matplotlib seaborn scikit-learn shap -y
```
### Exit this environment:
```
conda deactivate
```

## 🔑 Obtain the API keys for the third-party LLM

Create an Azure account [here](https://azure.microsoft.com/en-gb/pricing/purchase-options/azure-account?icid=azurefreeaccount).

Create an Azure OpenAI Service resource by following the steps in the [official guide](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/create-resource?pivots=web-portal) and this [user-friendly unofficial guide](https://ivanatilca.medium.com/a-step-by-step-guide-to-deploying-open-ai-models-on-microsoft-azure-cab86664fbb4).

At the Configure network security step, Allow all networks is easiest and should be acceptable for most users.

At the final “Deploy a model” step, we recommend selecting gpt-4, gpt-4o, or better (gpt-3.5 is not recommended due to poor performance). More specifically, please select one of these supported versions:
```
ALLOWED_MODELS = [
    # GPT-4o:
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    # GPT-4-turbo:
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    # GPT-3.5-turbo:
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
]
```

For instance: gpt-4o-2024-08-06 in supported versions list means model gpt-4 and model version 2024-08-06 in Azure.

Make note of the model type and version (e.g. gpt-4o-2024-08-06) you set up here as it is needed later.

When you are deploying the model, make note of the deployment name that you use as it is needed later.

In Azure OpenAI Studio:

Click the resource name at the top right of the screen to find: endpoint and key, make note of these as they are needed later. Azure info new UI

⚠️ Never share your API key with anyone and treat it as a “password”. A reminder to developers to to never commit your API keys to a public repository!


## 📈 Install the SynthCraft package

```bash
git clone https://github.com/vanderschaarlab/SynthCraft.git
```
Then navigate into the project folder and run:
```bash
conda activate SynthCraft
pip install -e .
```
Finally, you need to set up the configuration file for the LLM provider you chose.

Copy the Example .env file to the repo directory. Note that it should be placed directly inside the SynthCraft folder, not inside any subfolder; please see the end of this subsection below to check what your repo directory should contain at the end of the configuration process. On Windows you may wish to rename this file to keys.env to avoid the file being hidden / extension confusion.

Configure SynthCraft to work with the LLM provider you chose by following the appropiate instructions below:

### OpenAI
Open the `.env`/`keys.env` file in the repo directory and replace the value of

`OPENAI_API_KEY="API_KEY_FOR_OPENAI"`
with the key you obtained. Make sure to replace the text `API_KEY_FOR_OPENAI` but keep the quotes (e.g. if your key is abc123, the line should look like `OPENAI_API_KEY="abc123"`).

### AzureOpenAI
Open the `.env`/`keys.env` file in the repo directory.

`AZURE_OPENAI_API_KEY__my-endpoint-1="API_KEY_FOR_AZURE_ENDPOINT_1"`
Update the value `"API_KEY_FOR_AZURE_ENDPOINT_1"` with the API key you obtained.

Replace my-endpoint-1 template with the ID of the endpoint you are actually using. For example, if your endpoint is https://my-clinic.openai.azure.com/, use the my-clinic part. In this example case, the line would look like:

`AZURE_OPENAI_API_KEY__my-clinic="your actual API key"`
Copy the Example az_openai_config.yml file to the repo directory. Note that it should be placed directly inside the SynthCraft folder, not inside any subfolder; please see the end of this subsection below to check what your repo directory should contain at the end of the configuration process.

Open the az_openai_config.yml file in the repo directory:
```
models:
    - name: "your-custom-name"
    endpoint: "https://my-endpoint-1.openai.azure.com/"
    deployment_name: "your-deployment-name"
    api_version: "2024-02-01"
    model: "gpt-4-0125-preview"
    # Any lines with a '#' are comments and are just for your information.
```
The fields in this file are explained below. What you need to set the values to is shown in italics below. Note: you should not remove the quotation marks around the values.

**name:** This is used to identify the model in the UI. It will appear as config_item_name when you select the azure_openai_nextgen engine in 🗨️ Research Management. You can set this to any value you like.

**endpoint:** This identifies a URL needed to connect to the API. Set this to the endpoint you obtained.

**deployment_name:** This is the name of the deployment you set up in the Azure OpenAI Portal. Set this to the deployment name you obtained.

**api_version:** This is the version of the API protocol you are using. You can find the possible values for this field here (more info here) It is recommended to set this to the latest version available.

**model:** This needs to match the model type and version you set up in the Azure OpenAI Portal. It should be formatted exactly as the matching version from supported versions. For example, use gpt-4-0125-preview if you deployed a gpt-4 model and set the version to 0125-preview. This field should be set to model type and version from earlier.

At the end of the configuration, your repo directory should looks something like this:
```
project_directory/
├─ config_examples/
├─ docs/
├─ ...
├─ tests/
.coveragerc
.gitignore
...
.env  # Or, keys.env. The main configuration file.
az_openai_config.yml  # If using Azure OpenAI Service, its configuration file.
...
setup.py
tox.ini
```

## 🚀 Usage: SynthCraft

In order to run SynthCraft, launch SynthCraft UI, run the command:

`streamlit run entry/st/app.py`

This will show the output like:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.0.68:8501
```

### Start a new session
Navigate to the Local URL from the terminal output in your browser. We have tested SynthCraft on Google Chrome, so it is recommended for best UI compatibility.

Please select 🗨️ Research Management from the side panel. On this page, choose the *engine* to be:
```
openai_synthetic
```
or
```
azure_openai_synthetic
```
depending on the OpenAI model provider you are using.

From the Select engine dropdown, choose the engine that matches your LLM provider.

You may give a custom name to your session in the Session Name field, otherwise an auto-generated name will be assigned.

The Engine parameters section allows you to configure certain settings of the “engine”, such as the specific LLM model (when applicable). Hover over the  icon next to each setting to see a tooltip with more information.

Once you select Start new session, you will be taken to the main SynthCraft session screen.


## 🔏 Data Privacy
<!-- exclude_docs -->
> [!WARNING]  
> It is crucial to understand the data privacy and confidentiality implications of using SynthCraft. Please ensure to read and understand the in full prior to installing and using the system.
<!-- exclude_docs_end -->
<!-- include_docs
```{admonition} Warning
:class: attention

It is crucial to understand the data privacy and confidentiality implications of using SynthCraft. Please ensure to read and understand the [Data Privacy documentation section](dataprivacy.md) in full prior to installing and using the system.
```
include_docs_end -->

When using SynthCraft with real-world clinical data, you as the clinician scientist act as the data steward, and are responsible for ensuring that the use of the data complies with all the relevant laws and regulations, as well as ethical considerations. SynthCraft aims to provide a secure and privacy-preserving environment for data exploration and model building, while balancing this with leveraging the capabilities of the most advanced large language models (LLMs).

We provide a detailed section in the documentation, which summarizes the data privacy fundamentals of SynthCraft and should allow you to make an informed decision about using the system with your data. It is essential that you read and understand this prior to using SynthCraft, please find the link below:

<!-- exclude_docs -->
#### [📕 **Must-read:** Data Privacy documentation](https://github.com/vanderschaarlab/SynthCraft/blob/main/docs/dataprivacy.md)
<!-- exclude_docs_end -->
<!-- include_docs
#### [📕 **Must-read:** Data Privacy documentation](dataprivacy.md)
include_docs_end -->


