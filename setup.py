from setuptools import setup, find_packages

with open('requirements.txt','r', encoding='utf-8') as fh:
    requirements=[line.strip() for line in fh if line.strip() and not line.startswith('#')]
    
    setup(
        name='car_price_predictor',
        version='0.1.0',
        author='Bruno',
        authr_email='brunosharif89@gmail.com',
        description='A MLOPS project for car price prediction',
        long_description='An end to end MLOPS project for predicting car prices using structured vehicle data',
        long_description_content_type='text/markdown',
        packages=find_packages(where='src'),
        package_dir={'':'src'},
        python_requires='>=3.8',
        install_requires=requirements,
        classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
        
    )