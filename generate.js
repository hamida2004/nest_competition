// Define the intervals for each dataset
const datasetIntervals = {
    Bias_correction_ucl: {
      station: [1.0, 25.0],
      Present_Tmax: [20.0, 37.6],
      Present_Tmin: [11.3, 29.9],
      LDAPS_RHmin: [19.79466629, 98.5247345],
      LDAPS_RHmax: [58.93628311, 100.0001526],
      LDAPS_Tmax_lapse: [17.62495378, 38.54225522],
      LDAPS_Tmin_lapse: [14.27264631, 29.61934244],
      LDAPS_WS: [2.882579625, 21.85762099],
      LDAPS_LH: [-13.60321209, 213.4140062],
      LDAPS_CC1: [0.0, 0.967277328],
      LDAPS_CC2: [0.0, 0.96835306],
      LDAPS_CC3: [0.0, 0.983788755],
      LDAPS_CC4: [0.0, 0.974709524],
      LDAPS_PPT1: [0.0, 23.70154408],
      LDAPS_PPT2: [0.0, 21.62166078],
      LDAPS_PPT3: [0.0, 15.84123484],
      LDAPS_PPT4: [0.0, 16.65546921],
      lat: [37.4562, 37.645],
      lon: [126.826, 127.135],
      DEM: [12.37, 212.335],
      Slope: [0.0984746, 5.17823],
      "Solar radiation": [4329.520508, 5992.895996],
      Next_Tmax: [17.4, 38.9],
      Next_Tmin: [11.3, 29.8],
    },
    dataset_sans_nuls: {
      Longitude: [-8.655394, -8.490956],
      Latitude: [51.934883, 52.138499],
      Speed: [23, 114],
      CellID: [1, 4],
      RSRP: [-125, -88],
      RSRQ: [-16, -7],
      SNR: [-4.0, 25.0],
      CQI: [1, 15],
      RSSI: [-94, -67],
      DL_bitrate: [0, 56602],
      UL_bitrate: [0, 1780],
      NRxRSRP: [-126.0, -51.0],
      NRxRSRQ: [-225.0, -3.0],
      ServingCell_Lon: [-8.670811, -8.489758],
      ServingCell_Lat: [51.928426, 52.138878],
      ServingCell_Distance: [217.33, 10334.34],
    },
    dataset_with_failure_type: {
      Air_temperature: [295.3, 304.5],
      Process_temperature: [305.7, 313.8],
      Rotational_speed: [1168, 2886],
      Torque: [3.8, 76.6],
      Tool_wear: [0, 253],
    },
  };
  
  // Function to generate random values within an interval
  function getRandomValue(min, max) {
    return Math.random() * (max - min) + min;
  }
  
  // Function to generate data for a dataset over time
  function generateDatasetData(intervals, numMinutes) {
    const data = [];
  
    for (let minute = 1; minute <= numMinutes; minute++) {
      const entry = { minute };
  
      // Generate random values for each feature
      for (const [feature, [min, max]] of Object.entries(intervals)) {
        entry[feature] = getRandomValue(min, max);
      }
  
      data.push(entry);
    }
  
    return data;
  }
  
  // Generate data for each dataset
  const numMinutes = 60; // Generate data for 60 minutes (1 hour)
  const biasCorrectionData = generateDatasetData(datasetIntervals.Bias_correction_ucl, numMinutes);
  const datasetSansNulsData = generateDatasetData(datasetIntervals.dataset_sans_nuls, numMinutes);
  const datasetWithFailureTypeData = generateDatasetData(datasetIntervals.dataset_with_failure_type, numMinutes);
  
