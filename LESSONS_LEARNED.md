## Lessons learnt
 - Start with top 10 features from web and generate synthetic data based instead starting from data generation and finding features 
 - Most problems faced with synthetic data generation with random values instead of realistic data, generation should be formula driven like telematics_opt_in_rate, reporting_delay_std
 - Model will not learn if corelation of features are not right. So, granular corelation feature store validation is very important, including granular graph feature validation.
- Kaggle for practice and AWS for final inference deployment
- .env file copied from windows to wsl, not able to connect to neo4j due to invalid line endings
 use this command to fix all windows to linux and save troubleshooting time, git config --global core.autocrlf input