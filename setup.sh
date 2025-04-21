#!/bin/bash

echo_error(){
    echo "Error occurred in step: $1"
    exit 1
}

repo_checkout(){
    #Repos to checkout
    repos=(
        "https://github.com/open-mmlab/mmcv.git"
        # "https://github.com/ViTAE-Transformer/ViTPose.git"
        "https://github.com/shubham-goel/4D-Humans.git FDHumans"
        "https://github.com/facebookresearch/sapiens.git"
    )

    # Directory to clone repositories into
    clone_dir="./"

    # Create the directory if it doesn't exist
    mkdir -p "$clone_dir"

    # Navigate to the directory
    cd "$clone_dir" || { echo "Failed to enter directory $clone_dir"; exit 1; }

    # Function to clone a repository and optionally rename it
    clone_repo() {
        local repo_info="$1"
        local repo_url
        local custom_name

        # Split the repo_info into URL and custom name (if provided)
        repo_url=$(echo "$repo_info" | awk '{print $1}')
        custom_name=$(echo "$repo_info" | awk '{print $2}')

        # Extract the repo name from URL
        local repo_name
        repo_name=$(basename "$repo_url" .git)

        echo "Cloning $repo_name..."

        # Clone the repository
        if git clone "$repo_url"; then
            echo "Successfully cloned $repo_name."

            # Rename the repository directory if a custom name is provided
            if [[ -n "$custom_name" && "$custom_name" != "$repo_name" ]]; then
                mv "$repo_name" "$custom_name" && echo "Renamed $repo_name to $custom_name."
            fi
        else
            echo "Error cloning $repo_name. Skipping..."
        fi
    }

    # Loop through the repository list and clone each one
    for repo in "${repos[@]}"; do
        clone_repo "$repo"
    done

    echo "All repositories processed."
}

weights_checkout(){
    conda install curl -y
    rm -rf checkpoints
    mkdir -p checkpoints
    cd checkpoints
    curl -L -o vitpose_checkpoint.pth "https://62afda.dm.files.1drv.com/y4mBDiqHvl4ClkQbjljDfxZ35JemNwe-D-YlTuMfeya1BIR5tVP3cO26ntjrJkBL-2L8beSmOOPy7149gWRMkDqTZCPhS--XxryYZLSGtdKxR5ADq-9S_6ApoHxLbQP4MOs63iPz2jSLQMFqJFcFdoXZ2ml2HyvGkCu7MxyP9ELoZvtYRyipBDvsFvR2bN7xUknS6LR5HdBjGpZtM7saMmIXQ"
    cd ..
}

download_sapiens_models() {
    echo "Downloading Sapiens models..."
    
    # Create model directory
    mkdir -p models/sapiens
    cd models/sapiens
    
    # Download Sapiens models using curl
    # These URLs would need to be replaced with the actual download URLs for the models
    
    # 2B model
    echo "Downloading Sapiens 2B model..."
    if [ ! -f "sapiens_2b_coco_best_coco_AP_822_torchscript.pt2" ]; then
        curl -L -o sapiens_2b_coco_best_coco_AP_822_torchscript.pt2 "https://dl.fbaipublicfiles.com/sapien/sapiens_2b_coco_best_coco_AP_822_torchscript.pt2"
        echo "Downloaded Sapiens 2B model"
    else
        echo "Sapiens 2B model already exists, skipping download"
    fi
    
    # 1B model
    echo "Downloading Sapiens 1B model..."
    if [ ! -f "sapiens_1b_coco_best_coco_AP_821_torchscript.pt2" ]; then
        curl -L -o sapiens_1b_coco_best_coco_AP_821_torchscript.pt2 "https://dl.fbaipublicfiles.com/sapien/sapiens_1b_coco_best_coco_AP_821_torchscript.pt2"
        echo "Downloaded Sapiens 1B model"
    else
        echo "Sapiens 1B model already exists, skipping download"
    fi
    
    # 0.6B model
    echo "Downloading Sapiens 0.6B model..."
    if [ ! -f "sapiens_0.6b_coco_best_coco_AP_812_torchscript.pt2" ]; then
        curl -L -o sapiens_0.6b_coco_best_coco_AP_812_torchscript.pt2 "https://dl.fbaipublicfiles.com/sapien/sapiens_0.6b_coco_best_coco_AP_812_torchscript.pt2"
        echo "Downloaded Sapiens 0.6B model"
    else
        echo "Sapiens 0.6B model already exists, skipping download"
    fi
    
    # 0.3B model
    echo "Downloading Sapiens 0.3B model..."
    if [ ! -f "sapiens_0.3b_coco_best_coco_AP_796_torchscript.pt2" ]; then
        curl -L -o sapiens_0.3b_coco_best_coco_AP_796_torchscript.pt2 "https://dl.fbaipublicfiles.com/sapien/sapiens_0.3b_coco_best_coco_AP_796_torchscript.pt2"
        echo "Downloaded Sapiens 0.3B model"
    else
        echo "Sapiens 0.3B model already exists, skipping download"
    fi
    
    cd ../..
    echo "All Sapiens models downloaded successfully"
}

create_conda_env(){
    conda create --name poseapi python=3.10.13 -y
    eval "$(conda shell.bash hook)"
    conda activate poseapi
    conda install ipykernel -y
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
}

create_env_from_file(){
    conda env create -f environment.yml
    conda activate poseapi
}

mediapipe_setup(){
    pip install mediapipe
    pip install apex
}

vitpose_setup(){
    cd mmcv
    git checkout v1.3.9
    MMCV_WITH_OPS=1 pip install -e .
    cd ../ViTPose
    pip install -v -e .
    cd ..
    pip install timm==0.4.9 einops
}

verifiy_opengl(){
    if conda list | grep -q "osmesa"; then
        echo "osmesa is already installed in the current Conda environment."
    else
        echo "osmesa is not installed. Installing osmesa..."
        git clone https://github.com/mmatl/pyopengl.git
        pip install ./pyopengl
        # # Install osmesa in the current Conda environment
        # conda install -c conda-forge osmesa -y
        
        # if [ $? -eq 0 ]; then
        #     echo "osmesa successfully installed in the current Conda environment."
        # else
        #     echo "Failed to install osmesa. Please check your Conda environment and try again."
        #     exit 1
        # fi
    fi
}

fdhuman_setup(){
    cd FDHumans
    pip install -e .[all]
    cd ..
    verifiy_opengl || echo_error "opengl"
}

sapiens_setup(){
    echo "Setting up Sapiens..."
    
    # Navigate to the sapiens directory
    cd sapiens/lite
    
    # Install dependencies
    pip install -r requirements.txt
    
    # Install the package in development mode
    pip install -e .
    
    cd ../..
    
    echo "Sapiens setup completed successfully"
}

gradio_setup(){
    pip install -r requirements.txt
}

flask_setup(){
    echo "Setting up minimal Flask API environment..."
    
    # Install bare minimum Flask packages
    pip install flask flask-cors
    
    # Create basic directory structure
    mkdir -p uploads
    mkdir -p results
    
    echo "Minimal Flask API setup completed"
}

main(){
    echo "Starting setup process"
    repo_checkout || echo_error "repo_checkout"
    # weights_checkout || echo_error "weights" #sees like they blocked the download, do manually
    create_conda_env || echo_error "create_conda_env"
    gradio_setup || echo_error "Gradio"
    mediapipe_setup || echo_error "mediapipe"
    # # vitpose_setup || echo_error "ViTPose"
    fdhuman_setup || echo_error "FDHumans"
    download_sapiens_models || echo_error "Sapiens models download"
    sapiens_setup || echo_error "Sapiens"
    flask_setup || echo_error "Flask minimal setup"
    
    echo "Setup finished!"
    echo "To start developing the Flask API, activate the conda environment with 'conda activate poseapi'"
}

main