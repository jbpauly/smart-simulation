================
smart-simulation
================


.. image:: https://img.shields.io/pypi/v/smart_simulation.svg
        :target: https://pypi.python.org/pypi/smart_simulation

.. image:: https://img.shields.io/travis/jbpauly/smart_simulation.svg
        :target: https://travis-ci.com/jbpauly/smart_simulation

.. image:: https://readthedocs.org/projects/smart-simulation/badge/?version=latest
        :target: https://smart-simulation.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/jbpauly/smart_simulation/shield.svg
     :target: https://pyup.io/repos/github/jbpauly/smart_simulation/
     :alt: Updates


Simulator for Smart Subscription Services.

This package generates data, simulations, and counter-factual scenarios for weight-based smart subscriptions.
Weight-based smart subscriptions utilize hardware (wifi weight scales) and software (machine learning) to provide
just-in-time delivery of repeatably purchased goods. Bottomless is the first of its kind.

How does a weight based subscription work? As an example, the Bottomless service will be described. Bottomless provides
customers with a wifi scale linked to their account. Customers select a delivery preference from three choices: Never
Run Out, Just-In-Time, As Fresh As Possible. This preference provides additional context for the ordering algorithm.
Customers then store their subscribed product, like coffee beans, on the scale. Consumption data is captured
by weight measurements and monitored by Bottomless' ordering algorithm. Orders are then triggered to arrive when
expected by the customer based off their resupply preference.

Data and machine learning is at the core of smart subscriptions. Starting or advancing a smart subscription business
requires a quality dataset. There are multiple agents or factors involved with the working of smart subscriptions:
*Customers, Scales, Product Marketplace, Warehouses, Shipping*.

Examples of when the package could be used:
 - Generate a dataset to create a just-in-time ordering algorithm
 - Explore impact of real world scenarios or changes to the underlying algorithm on a subscription service
    - Answer questions like
        - "How much will ordering accuracy decrease with a 30% reduction in USPS shipping
          estimates confidence?"
        - "How much would monthly revenue drop if a bulk shipment of a product is delayed
          and 5% of customers buy from the grocery store?"
        - "How many additional shipments could be bundled if the time buffers on the ordering strategies for
          non-perishable goods are extended by 1 day? 2 days? 3 days?"
    - Test ordering algorithms with unique goals
        - Maximize bundled shipments
        - Maximize attainment of order preferences


Dependencies
------------
Python >= 3.7


Run Locally in Virtual Environment
----------------------------------
Documentation example uses ``bash commands`` and works for Mac and Linux OS.

Clone Repository to Local Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Detailed instructions on cloning:
https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository

``git clone https://github.com/jbpauly/smart-simulation.git``

Move to The Project Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``cd smart-simulation``

Create the Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``python -m venv venv``

Activate the Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``source venv/bin/activate``

Install Requirements
^^^^^^^^^^^^^^^^^^^^
``pip install -r requirements.txt``

Install the Package
^^^^^^^^^^^^^^^^^^^
``pip install -e .``

Run the Package Locally
^^^^^^^^^^^^^^^^^^^^^^^

**Explore CLI Options**

``smart_simulation --help``

``smart_simulation create-consumption --help`` ::

    Usage: smart_simulation create-consumption [OPTIONS]

      Create consumer consumption data (servings) and associated scale data
      (weight)

    Options:
      -c, --customer_number [0]
      -d, --days INTEGER
      -s, --start_date TEXT
      -f, --file_name TEXT
      -p, --output_path PATH
      --help                     Show this message and exit.


Example Use
^^^^^^^^^^^
``smart_simulation create-consumption`` ::

    Customer number (0): 0
    Days: 100
    Start date [2020-01-01]: <enter to use default>
    File name (without file type extension): customer_data
    Output path [/<your local directory path>/smart-simulation/smart_simulation/outputs]: <enter to use default>

**Check output path for customer_data.csv**

.. csv-table:: Customer Data
   :header: Index, date, servings, weight
   :widths: 10, 10, 10, 10
   :stub-columns: 1

   0, 2020-01-01, 1, 13.0
   1, 2020-01-02, 1, 12.61
   2,2020-01-03, 1, 12.22
   3,2020-01-04, 0, 12.22


Road Map
--------

Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
