# YoloStudio

<p align="center">
  <img src="docs/assets/github-banner.png" alt="YoloStudio — Local YOLO detection workbench" width="100%">
</p>

<p align="center">
  <strong>Local YOLO detection workbench · 本地目标检测工作台</strong><br>
  Annotate · Freeze · Train · Review　｜　标注 · 冻结 · 训练 · 复核
</p>

[English](#english) · [中文](#中文)

<a id="english"></a>

## English

YoloStudio is a local workbench for object detection. It takes a folder or zip of images through class definition, bounding-box annotation, dataset freeze, Ultralytics training, visual review, and model export. Images, labels, runs, and exports stay in a workspace on the machine that runs the app.

The interface is in Chinese. Two roles share one database: an administrator runs the full workflow, and annotators label images in datasets an administrator has already imported.

This repository ships the application only. Bring your own images. Dataset files, the SQLite database, and trained weights are written under `workspace/` and are not part of the source tree.

### Requirements

- Python 3.11 or newer
- Node.js 18 or newer
- A local dataset: a folder of images, or a `.zip` of that folder

Training uses [Ultralytics](https://github.com/ultralytics/ultralytics). The backend install pulls in PyTorch. CUDA is selected when it is available, then Apple MPS, then CPU. The first training run also downloads the chosen pretrained weights, such as `yolov8n.pt`.

### Run it locally

Use two terminals. Start the API first.

**API**

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The first install is large because of PyTorch. When the process is up, the workspace directory and a SQLite database are created beside the repository.

```bash
curl http://127.0.0.1:8000/api/health
```

A healthy process returns `"status": "ok"` and the device it will train on.

**Interface**

```bash
cd frontend
npm install
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). The dev server proxies `/api` to `127.0.0.1:8000`.

On a fresh database the sign-in is:

| | |
| --- | --- |
| Username | `admin` |
| Password | `admin123` |

Change that password from **用户管理** before any other person can reach the machine. To choose the initial account yourself, set the variables below before the first launch. They apply only when the user table is still empty.

### First session

1. Sign in as the administrator.
2. On **项目与数据**, choose a local zip or folder, scan it, then import it. You can also upload a zip from the browser. Set the project name and dataset name before import. Reusing a project name keeps the same class library.
3. On **类别管理**, confirm the classes. Filename patterns can suggest classes; you still decide what is created.
4. On **图像标注**, draw boxes, adjust them, and save.
5. On **质量与版本**, read the quality summary and create a dataset version. The version is a frozen YOLO layout with an 80/10/10 split.
6. On **模型训练**, start from a preset such as `yolov8n.pt`. The run log updates while training is in progress.
7. On **效果预览** and **评估与导出**, inspect predictions and export `.pt` or ONNX from a finished run.

An annotator account, created under **用户管理**, can load a saved dataset and label it. Training, preview, evaluation, and user administration stay with the administrator.

### Single process

After the interface has been built, the API can serve it:

```bash
cd frontend && npm install && npm run build
cd ../backend && source .venv/bin/activate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

### Configuration

Settings use the `YOLOSTUDIO_` prefix.

| Variable | Default | Purpose |
| --- | --- | --- |
| `YOLOSTUDIO_WORKSPACE_ROOT` | `<repo>/workspace` | Database, imported images, versions, runs, exports |
| `YOLOSTUDIO_JWT_SECRET` | a development placeholder | Signing key for session tokens |
| `YOLOSTUDIO_DEFAULT_ADMIN_USERNAME` | `admin` | First administrator, created once |
| `YOLOSTUDIO_DEFAULT_ADMIN_PASSWORD` | `admin123` | Password for that first account |
| `YOLOSTUDIO_ACCESS_TOKEN_EXPIRE_MINUTES` | 7 days | Session lifetime |

Example:

```bash
export YOLOSTUDIO_JWT_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export YOLOSTUDIO_DEFAULT_ADMIN_PASSWORD='choose-a-long-password'
```

### Layout

```text
backend/     FastAPI service, training jobs, SQLite models
frontend/    React interface
workspace/   runtime data, created on startup, ignored by git
```

Versions land in `workspace/projects/<project_id>/versions/<version_id>/` with `images/`, `labels/`, and `data.yaml`. Runs land in `workspace/projects/<project_id>/runs/<run_id>/`.

### Development

```bash
cd backend && source .venv/bin/activate && python -m pytest -q
cd frontend && npm test -- --run && npm run build
```

### Security

YoloStudio is a single-machine tool. The development password and signing key are there so a checkout can boot. Replace both before the port is reachable by anyone else. Do not commit `workspace/`, datasets, or weight files.

---

<a id="中文"></a>

## 中文

YoloStudio 是一套本地目标检测工作台。它把图像目录或压缩包一路送到类别定义、边界框标注、数据版本冻结、Ultralytics 训练、结果复查和模型导出。图像、标注、训练记录和导出文件都留在运行这套程序的那台机器上。

界面语言是中文。同一套数据库里有两种角色：管理员走完整流程，标注员只在管理员已经导入的数据集上标注。

本仓库只包含程序。图像请自备。数据集、SQLite 数据库和训练权重写在 `workspace/` 下，不进入源码树。

### 环境

- Python 3.11 或更高版本
- Node.js 18 或更高版本
- 一份本地数据：图像文件夹，或该文件夹的 `.zip`

训练走 [Ultralytics](https://github.com/ultralytics/ultralytics)。安装后端时会一并安装 PyTorch。设备按 CUDA、Apple MPS、CPU 的顺序选用。第一次训练还会下载所选的预训练权重，例如 `yolov8n.pt`。

### 本地启动

开两个终端，先启动 API。

**API**

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows：.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

首次安装会比较久，时间主要花在 PyTorch。进程起来之后，会在仓库旁边创建 `workspace/` 和 SQLite 数据库。

```bash
curl http://127.0.0.1:8000/api/health
```

正常时返回 `"status": "ok"`，并带上即将用于训练的设备。

**界面**

```bash
cd frontend
npm install
npm run dev
```

打开 [http://127.0.0.1:5173](http://127.0.0.1:5173)。开发服务器把 `/api` 转发到 `127.0.0.1:8000`。

空数据库的初始登录是：

| | |
| --- | --- |
| 用户名 | `admin` |
| 密码 | `admin123` |

只要还有别人能访问这台机器，就先到 **用户管理** 里改掉这个密码。如果想自己指定初始账号，在第一次启动前设置下面的环境变量。用户表里已经有人之后，这些变量不会再创建账号。

### 第一次使用

1. 用管理员账号登录。
2. 在 **项目与数据** 里选择本地 zip 或文件夹，先扫描再导入。也可以直接在浏览器里上传 zip。导入前写好项目名和数据集名。项目名相同，类别库就沿用同一套。
3. 在 **类别管理** 里确认类别。文件名可以给出建议，最终是否创建由你决定。
4. 在 **图像标注** 里画框、调整并保存。
5. 在 **质量与版本** 里查看质量摘要，再生成数据版本。版本是一份冻结的 YOLO 目录，划分比例为 80/10/10。
6. 在 **模型训练** 里从一个预设权重开始，例如 `yolov8n.pt`。训练过程中日志会持续更新。
7. 在 **效果预览** 和 **评估与导出** 里查看预测，并从已完成的训练导出 `.pt` 或 ONNX。

在 **用户管理** 里创建的标注员可以加载已保存的数据集并标注。训练、预览、评估和用户管理仍由管理员操作。

### 单进程运行

界面构建完成后，可以由 API 直接提供页面：

```bash
cd frontend && npm install && npm run build
cd ../backend && source .venv/bin/activate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。

### 配置

配置项使用 `YOLOSTUDIO_` 前缀。

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `YOLOSTUDIO_WORKSPACE_ROOT` | `<仓库>/workspace` | 数据库、导入图像、版本、训练和导出 |
| `YOLOSTUDIO_JWT_SECRET` | 开发用占位密钥 | 登录令牌的签名密钥 |
| `YOLOSTUDIO_DEFAULT_ADMIN_USERNAME` | `admin` | 首次创建的管理员，只创建一次 |
| `YOLOSTUDIO_DEFAULT_ADMIN_PASSWORD` | `admin123` | 该账号的初始密码 |
| `YOLOSTUDIO_ACCESS_TOKEN_EXPIRE_MINUTES` | 7 天 | 登录有效期 |

示例：

```bash
export YOLOSTUDIO_JWT_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export YOLOSTUDIO_DEFAULT_ADMIN_PASSWORD='choose-a-long-password'
```

### 目录

```text
backend/     FastAPI 服务、训练任务、SQLite 模型
frontend/    React 界面
workspace/   运行时数据，启动时创建，已被 git 忽略
```

数据版本写在 `workspace/projects/<project_id>/versions/<version_id>/`，其中包含 `images/`、`labels/` 和 `data.yaml`。训练记录写在 `workspace/projects/<project_id>/runs/<run_id>/`。

### 开发

```bash
cd backend && source .venv/bin/activate && python -m pytest -q
cd frontend && npm test -- --run && npm run build
```

### 安全

YoloStudio 面向单机使用。开发用的密码和签名密钥是为了让一份新检出的代码能够直接启动。端口对其他人可达之前，请换掉这两项。不要把 `workspace/`、数据集或权重文件提交进仓库。
