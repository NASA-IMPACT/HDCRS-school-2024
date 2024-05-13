## Prerequisites

### High Performance Computing access

### Resources
1. https://judoor.fz-juelich.de/
2. https://nasa-impact.awsapps.com/start/
3. 

### Cloud environment access
1. Get your credentials and other information using https://creds-workshop.nasa-impact.net/
![Get Credentials](../images/credential.png)
![Credentials](../images/credentials-show.png)
2. Navigate to https://nasa-impact.awsapps.com/start#/ 
![Login Page](../images/login-1.png)
3. Log in using the credential provided
![Login with username and password](../images/login-2.png)
4. Navigate to the `Applications` tab
![Logged in home page](../images/loggedin.png)
5. Click and open `Amazon SageMaker Studio`
![List of Applications](../images/applications.png)
6. Once the Studio starts, Click on JupyterLab
![Sagemaker studio](../images/sagemaker-studio.png)
![JupyterLab spaces](../images/jupyterlab-spaces.png)
7. Click `Create JupyterLab Space`
![JupyterLab spaces](../images/create-jupyterlab-env.png)
8. Give it a name. Eg: `Workshop`
9. Once initialized, change Instance type to `ml.t3.large` and storage to `50`
![Change instance type](../images/update-instance-type.png)
10. Click on `Run Space`. If it throws an error, you might have to pick an Image. The top setting called `Latest` works. 
![Run space](../images/updated-instance-config.png)

## HLS fine-tuning
Harmonized Landsat and Sentinel - 2 (HLS) Foundation model (Prithvi) is currently available in Huggingface(https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M). In this hands-on, finetuning will be done in the Julich Supercomputing Center (JSC). Please follow instructions listed in the [notebook](../HLS-finetuning/notebooks/hls-fm-finteuning.ipynb).

**Note: Before we start, please clone this repository in the JSC notebook environment.**

### Prepare Environment for Fine-tuning
 Clone this repository `git clone https://github.com/nasa-impact/HDCRS-school-2024.git`
```
a. Click `git`
b. Click on `Git Clone Repo`
![Git clone](../images/git-clone-1.png)
c. Paste `https://github.com/nasa-impact/HDCRS-school-2024.git` and Click on `Clone`.
![Cloned repository](../images/smd-hls-git-clone.png)
![Cloned repository](../images/smd-hls-cloned-content.png)
```

