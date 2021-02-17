from setuptools import setup, find_packages

setup(
    # Metadata
    name="DRL_Homework2",
    version=0.1,
    author=("Tolga Ok, Emircan Kilickaya, Batuhan Ince,"
            " Kubilay Kagan Komurcu & Nazim Kemal Ure"),
    author_email="okt@itu.edu.tr",
    url="https://ninova.itu.edu.tr/Sinif/11681.42666",
    description="HW2",
    long_description="DQN & RAINBOW",
    license="MIT",

    # Package info
    packages=["dqn", ],
    zip_safe=False,
)
