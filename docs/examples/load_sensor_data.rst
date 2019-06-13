.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_load_sensor_data.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_load_sensor_data.py:


Single Dataset
==============

A simple example on how to work with a single Dataset.



.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /examples/images/sphx_glr_load_sensor_data_001.png
            :class: sphx-glr-multi-img

    *

      .. image:: /examples/images/sphx_glr_load_sensor_data_002.png
            :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Sensor ID: 7fad
    Start Date (UTC): 2019-04-30 07:33:12
    Stop Date (UTC): 2019-04-30 07:33:59
    Enabled Sensors: ('gyro', 'acc')
    Analog is disabled: True
    The acc recordings are 3D and have the length 9597
    The new datastream has a length of 4799
    Acc has now a length of 4799
    Gyro has now a length of 4799
    The old and the new are identical: True
                  gyr_x     gyr_y     gyr_z     acc_x     acc_y     acc_z
    n_samples                                                            
    0          0.304595  0.243573 -0.118592  0.049722  0.029840  1.005433
    1         -0.410535 -0.219459 -0.317742  0.054190  0.031612  1.007515
    2         -0.038394 -0.003786 -0.160372  0.052292  0.033968  1.006887
    3         -0.210249 -0.059454  0.068967  0.061097  0.035140  1.006755
    4         -0.087388 -0.237385 -0.103119  0.064567  0.033970  1.007098




|


.. code-block:: default


    import matplotlib.pyplot as plt
    from pathlib import Path

    from NilsPodLib import Dataset

    FILEPATH = Path('../tests/test_data/synced_sample_session/NilsPodX-7FAD_20190430_0933.bin')

    # Create a Dataset Object from the bin file
    dataset = Dataset.from_bin_file(FILEPATH)

    # You can access the metainformation about your dataset using the `info` attr.
    # For a full list of available attributes see NilsPodLib.header.HeaderFields
    print('Sensor ID:', dataset.info.sensor_id)
    print('Start Date (UTC):', dataset.info.utc_datetime_start)
    print('Stop Date (UTC):', dataset.info.utc_datetime_stop)
    print('Enabled Sensors:', dataset.info.enabled_sensors)

    # You can access the individual sensor data directly from the dataset object using the names provided
    # in dataset.info.enabled_sensors
    datastream_acc = dataset.acc

    # If a sensor is disabled, this will return `None`
    print('Analog is disabled:', dataset.analog is None)

    # Access the data of a datastream object as a numpy.array using the `data` attribute
    print('The acc recordings are {}D and have the length {}'.format(*datastream_acc.data.T.shape))

    # Convenience methods are available for common operations. E.g. Norm or downsample
    plt.figure()
    plt.title('Acc Norm')
    plt.plot(datastream_acc.norm())
    plt.show()

    downsampled_datastream = datastream_acc.downsample(factor=2)
    print('The new datastream has a length of', len(downsampled_datastream.data))

    # However, for many operations it makes more sense to apply them to the Dataset instead of the Datastream.
    # This will apply the operations to all Datastream and return a new Dataset object

    downsampled_dataset = dataset.downsample(factor=2)
    print('Acc has now a length of', len(downsampled_dataset.acc.data))
    print('Gyro has now a length of', len(downsampled_dataset.gyro.data))

    # By default this returns a copy of the dataset and all datastreams. If this is a performance concern, the dataset can
    # be modified inplace:

    downsampled_dataset = dataset.downsample(factor=2, inplace=True)
    print('The old and the new are identical:', id(dataset) == id(downsampled_dataset))

    # Usually, before using any data it needs to be calibrated. The dataset object offers factory_calibrations for all
    # important sensors. These convert the datastreams into physical units

    dataset_cal = dataset.factory_calibrate_baro()
    # For acc and gyro a convenience method is provided.
    dataset_cal = dataset_cal.factory_calibrate_imu()

    # However, for more precise measurements an actual IMU Calibration using the `calibrate_{acc,gyro,imu}` methods should
    # be performed.

    # After calibration and initial operations on all datastreams, the easiest way to interface with further processing
    # pipelines is a conversion into a pandas DataFrame

    df = dataset_cal.data_as_df()

    print(df.head())

    df.plot()
    plt.show()


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.474 seconds)


.. _sphx_glr_download_examples_load_sensor_data.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: load_sensor_data.py <load_sensor_data.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: load_sensor_data.ipynb <load_sensor_data.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
