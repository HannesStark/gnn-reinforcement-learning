### Copy files to your VM from your local machine.
You can use the gcloud tool to upload files to your machine.

`gcloud compute scp --project tum-adlr-ws21-04 --zone europe-west3-b --recurse <local file or directory> pytorch-workstation-01-vm:~/`


### Access the running Jupyter notebook.
We've already started a Jupyter notebook instance on the VM for your convenience. In order to get link that can be used to access Jupyter Lab run the following command.

`gcloud compute instances describe --project tum-adlr-ws21-04 --zone europe-west3-b pytorch-workstation-01-vm | grep googleusercontent.com | grep datalab




### Setting up remote desktop acces with GUI for GCP VM

Helpful guides:

For Ubuntu: 
[Using OpenAI Gym and Pybullet at Google Cloud Compute Engine with Graphical User Interface](https://medium.com/@yazarmusa/using-openai-gym-and-pybullet-at-google-cloud-compute-engines-with-graphical-user-interface-for-233b91375f0e)
For Debian: 
[Your desktop on Google Cloud Platform](https://medium.com/google-cloud/linux-gui-on-the-google-cloud-platform-800719ab27c5)


