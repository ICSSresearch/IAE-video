from setuptools import setup    # , find_packages
import os

# custom made numba, threadhandler

# print(find_packages())

s = setup(name='bundled',
          version='1.0',
          description='To stablize videos using bundled path method',
          url='',
          author='Celine Wei',
          author_email='cw.git.general@gmail.com',
          license='MIT',
          packages=['bundled'],
          install_requires=['tqdm>=4.3.1', 'numpy>=1.16.2', 'joblib>=0.13.2', 'sharedmem>=0.3.5', 'matplotlib>=2.2.2',
                            'threadhandler>=1.0', 'numba==0.47.0iae',
                            'opencv-python==3.4.9.33', 'opencv-contrib-python==3.4.9.33', 'llvmlite==0.31', 'scipy>=0.16'],
          dependency_links=['https://github.com/celinew1221/threadhandler/tarball/master#egg=threadhandler-v1.0',
                            'https://github.com/celinew1221/numba/tarball/0.47.0iae#egg=numba-0.47.0iae'],
          zip_safe=False,
          package_dir={'bundled': 'bundled/'},
          package_data={'./': ['asap.so', 'meshhomo.so', 'meshwarp.so', 'bundled/asap.so', 'bundled/meshhomo.so',
                               'bundled/meshwarp.so']},)

installation_path = s.command_obj['install'].install_lib
bundled_folder_name = os.popen('ls %s | grep bundled' % installation_path).read().replace("\n", "")
bundled_path = "{:s}{:s}/{:s}".format(installation_path, bundled_folder_name, "bundled")
print("\nMoving dynamic library to %s" % bundled_path)
os.system("cp bundled/*.so %s" % bundled_path)
