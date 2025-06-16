from setuptools import find_packages, setup

package_name = 'go_to_rescuepoint'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shs',
    maintainer_email='shs102030405060@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rescuepoint_follower = go_to_rescuepoint.rescuepoint_follower:main',
            'rescuepoint_follower2 = go_to_rescuepoint.rescuepoint_follower2:main',
            'rescuepoint_server = go_to_rescuepoint.rescuepoint_server:main',

        ],
    },
)
