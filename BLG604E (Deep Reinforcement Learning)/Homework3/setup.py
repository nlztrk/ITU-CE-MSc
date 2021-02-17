from setuptools import setup, find_packages

setup(
    # Metadata
    name="DRL_Homework3",
    version=0.1,
    author=("Tolga Ok, Emircan Kilickaya, Batuhan Ince,"
            " Kubilay Kagan Komurcu & Nazim Kemal Ure"),
    author_email="okt@itu.edu.tr",
    url="https://ninova.itu.edu.tr/Sinif/11681.42666",
    description="HW3 Policy Gradients",
    long_description="REINFORCE & A2C implementations",
    license="MIT",

    # Package info
    packages=["pg", ],
    zip_safe=False,
)
