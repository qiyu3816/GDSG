# Dataset for multi-server multi-user offloading problem

## Numerical range

Regarding the bandwidth setting of the OMA mechanism, many routers now use the 802.11ac protocol, where the bandwidth can be as high as 80MHz, and the common ones are 20MHz, 60MHz, etc. Based on previous dataset assumptions, 0.5MH and 20MHz were tried, and the calculated uplink rate did not exceed 1Mbps, which is not realistic at all (even using a 2019 4G mobile phone to connect to WiFi, the average large file upload rate is as high as 2.6MB /s≈21.8Mbps), the most serious thing is that this will limit the input data size of the computing task. We can only set a very small task data size to avoid excessive transmission costs, but this is actually contradictory to the assumption of offloading of computationally intensive tasks. Therefore, the bandwidth is set to **80MHz**, so that the transmission rate is basically above 1Mbps. Specifically, the size of task input should be satisfy that in most cases, local execution will timeout, while offloaded execution can reach the latency requirement: 
$$
\frac{s}{r_u}+\frac{3\times10^3\times s}{F_t}<1.8,s<1.14\times10^7 \\
\frac{3\times10^3\times s}{f_l}<2,s<5.4\times10^6
$$
From the above results, we select the interval. 

For the local computation resources of mobile devices, taking the Kirin 980 of the 2019 mobile phone as a reference, the maximum is 2\*2.6+2\*1.92+4*1.8=16.22GHz. However, mobile devices are limited by available power (the maximum operating power is commonly 3W, generally 1~2W, which is the total power of all tasks of the device) and other computing tasks, and it is basically impossible to run the chip at its theoretical frequency. So we set a mean of 7.5GHz and a variance of 5, a lower bound of 1GHz and an upper bound of 10GHz. 

In summary, the current dataset hyperparameter settings are: 

```python
	F_t = 33.6e9
    theta = 2.0
    P_t = 0.3
    P_I = 0.15
    kappa = 1e-28
    B = 8e7
    N0 = 7.96159e-13  # xi**2, for SINR calculation

    is_mu, is_sigma, is_low, is_up = 0.65e7, 0.3e7, 0.2e7, 1.1e7
    fl_mu, fl_sigma, fl_low, fl_up = 6.4e9, 5e9, 1e9, 10e9
```

For the construction of the topology map, the range of node_num corresponding to server_num is [server_num + 1, 4 * server_num - 2] to ensure density. In terms of connectivity, a user has a high probability of being connected to 2 servers, and a small probability of being connected to 1, 3, or 4 servers. Each server is guaranteed to be connected to at least one user. The current settings are fixed as server_num=4, node_num=12.

## Dataset format

| Attribute | Meaning                                                      | Shape                  |
| --------- | ------------------------------------------------------------ | ---------------------- |
| node      | node type, 0-1                                               | (server_num+user_num,) |
| edge      | edge index connectivity                                      | (K, 2)                 |
| node_raw  | (input_size, required_cycles, f_locals, alphas) of each user | (user_num * 4, )       |
| edge_raw  | channel_gain of each edge                                    | (K, )                  |
| edge_attr | (local_cost, trans_cost, offload_cost, least_ratio, local_sat) of each edge | (edge_num * 5, )       |
| gt_edges  | ground truth edges                                           | (k, 2)                 |
| gt_ws     | ground truth weights that indicate the resource allocation ratios | (k, )                  |
| gt_cost   | ground truth cost                                            | ()                     |

## Dataset analysis

This table shows the numerical range of the key features of each dataset, where the last column "extreme_trans" represents the ratio of the extremely great transmission cost (trans_cost > the maximum gt_cost of the current dataset): 

| **Dataset ID** | **local_cost**          | **trans_cost**                 | **offload_cost**        | **gt_cost**            | **extreme_trans**  |
|---------------|-------------------------|-------------------------------|-------------------------|------------------------|--------------------|
| 3s6u_2000     | (0.0109, 25.7008)       | (0.0091, 2.1783e+08)           | (0.0289, 0.9704)        | (1.2624, 19.3490)      | 872/(2000*6)       |
| 3s6u_60000    | (0.0018, 31.8134)       | (0.0078, 7.5559e+09)           | (0.0272, 0.9773)        | (0.8049, 35.5029)      | 26786/(60000*6)    |
| 3s8u_2000     | (0.0053, 28.7219)       | (0.0136, 272461376.0000)       | (0.0309, 0.9732)        | (2.5157, 25.3149)      | 1298/(2000*8)      |
| 3s8u_60000    | (0.0011, 31.4758)       | (0.0089, 219211268096.0000)    | (0.0273, 0.9816)        | (1.6544, 31.5801)      | 35152/(60000*8)    |
| 4s10u_10000   | (0.0017, 31.6464)       | (0.0100, 27122554880.0000)     | (0.0275, 0.9781)        | (2.7445, 33.6241)      | 8392/(10000*10)    |
| 4s10u_80000   | (0.0018, 31.8686)       | (0.0076, 2.7769e+13)           | (0.0270, 0.9793)        | (2.1827, 32.9737)      | 66921/(80000*10)   |
| 4s12u_10000   | (0.0018, 30.2995)       | (0.0090, 3759381248.0000)      | (0.0279, 0.9774)        | (4.2492, 42.0025)      | 9481/(10000*12)    |
| 4s12u_80000   | (0.0012, 31.7184)       | (0.0088, 323389423616.0000)    | (0.0271, 0.9819)        | (4.1129, 44.2700)      | 76093/(80000*12)   |
| 7s24u_2000    | (0.0017, 28.5101)       | (0.0137, 32282272.0000)        | (0.0275, 0.9665)        | (15.1792, 57.4610)     | 1942/(2000*24)     |
| 7s24u_80000   | (0.0011, 32.2308)       | (0.0084, 1.5952e+14)           | (0.0270, 0.9814)        | (11.6166, 64.8265)     | 78614/(80000*24)   |
| 7s27u_2000    | (0.0020, 31.7782)       | (0.0120, 2221645824.0000)      | (0.0275, 0.9799)        | (16.8230, 69.7700)     | 2215/(2000*27)     |
| 7s27u_80000   | (0.0011, 31.9900)       | (0.0091, 31922281512960.0000)  | (0.0271, 0.9803)        | (15.4948, 75.3571)     | 87483/(80000*27)   |
| 10s31u_2000   | (0.0019, 30.7279)       | (0.0099, 207873952.0000)       | (0.0271, 0.9747)        | (19.7498, 61.8615)     | 1637/(2000*31)     |
| 10s31u_80000  | (0.0007, 31.6660)       | (0.0081, 55637422080.0000)     | (0.0269, 0.9813)        | (16.9502, 73.9701)     | 63171/(80000*31)   |
| 10s36u_2000   | (0.0026, 29.9773)       | (0.0110, 48567660.0000)        | (0.0291, 0.9755)        | (26.8321, 75.3846)     | 1827/(2000*36)     |
| 10s36u_80000  | (0.0010, 31.4044)       | (0.0094, 7.0247e+13)           | (0.0271, 0.9806)        | (21.6354, 88.8364)     | 72807/(80000*36)   |
| 20s61u_2000   | (0.0020, 28.2578)       | (0.0118, 249340035072.0000)    | (0.0272, 0.9644)        | (47.1416, 112.4311)    | 1186/(2000*61)     |
| 20s61u_80000  | (0.0007, 31.6702)       | (0.0081, 825527631872.0000)    | (0.0270, 0.9814)        | (42.6974, 124.0135)    | 47747/(80000*61)   |
| 20s68u_2000   | (0.0036, 31.3235)       | (0.0079, 622156800.0000)       | (0.0280, 0.9739)        | (54.6612, 133.8146)    | 1242/(2000*68)     |
| 20s68u_80000  | (0.0006, 32.5364)       | (0.0078, 338845171712.0000)    | (0.0270, 0.9805)        | (49.7855, 147.1985)    | 51677/(80000*68)   |

This table analyzes each truth testset. The second column represents the ratio of the total cost of the truth testset to the total cost of the suboptimal set, and the third column is the opposite (this column is equal to the exceed_ratio of the HEU method): 

| **name**      | **gt/lq**  | **lq/gt**  |
|---------------|------------|------------|
| r1000_3s6u    | 0.82       | 1.22       |
| r1000_3s8u    | 0.85       | 1.17       |
| r1000_4s10u   | 0.83       | 1.20       |
| r1000_4s12u   | 0.85       | 1.17       |
| r200_7s24u    | 0.8796     | 1.1368     |
| r200_7s27u    | 0.8731     | 1.1453     |
| r200_10s31u   | 0.8767     | 1.1406     |
| r200_10s36u   | 0.8624     | 1.1596     |
| r100_20s61u   | 0.8683     | 1.1516     |
| r100_20s68u   | 0.8449     | 1.1835     |
