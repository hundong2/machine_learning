# autogen

## microsoft study 


[ ] [ML Workflow](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/architecture/baseline-azure-ai-foundry-chat)
[ ] [Autogen official site](https://microsoft.github.io/autogen/stable/)
[ ] [autogen framework](https://microsoft.github.io/autogen/stable/). 
 - [ ] [guide for autogen studio](https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/installation.html). 

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### frontend install 

- path information : `autogen/python/packages/autogen-studio/frontend`. 

```sh
npm install -g gatsby-cli
npm install --global yarn
cd frontend
yarn install
yarn build
# Windows users may need alternative commands to build the frontend:
gatsby clean && rmdir /s /q ..\\autogenstudio\\web\\ui 2>nul & (set \"PREFIX_PATH_VALUE=\" || ver>nul) && gatsby build --prefix-paths && xcopy /E /I /Y public ..\\autogenstudio\\web\\ui

```
