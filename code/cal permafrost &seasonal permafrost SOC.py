import os
import numpy as np
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def WriteTifImg(filename, im_proj, im_geotrans, im_data, datatype=None, nodata=None):
    if datatype is None:
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    if nodata is not None:
        im_data = np.where(np.isnan(im_data), nodata, im_data)

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    if im_bands == 1:
        band = dataset.GetRasterBand(1)
        band.WriteArray(im_data)
        if nodata is not None:
            band.SetNoDataValue(nodata)
        band.FlushCache()
    else:
        for i in range(im_bands):
            band = dataset.GetRasterBand(i + 1)
            band.WriteArray(im_data[i])
            if nodata is not None:
                band.SetNoDataValue(nodata)
            band.FlushCache()

    dataset = None

def SOC_MULTIPLY_PERMAFROST(SOC_img, permafrost_img):
    for i in range(SOC_img.shape[0]):
        for j in range(SOC_img.shape[1]):
            if SOC_img[i][j]==np.nan and permafrost_img[i][j]==-9999:
                continue
            else:
                SOC_img[i][j]*=permafrost_img[i][j]
    return SOC_img

def mask_soc(soc_arr, mask_arr):
    mask = mask_arr.astype(np.float32)
    mask = np.where(mask == -9999, np.nan, mask)      # 掩膜 nodata → NaN
    return np.where(mask == 1, soc_arr, np.nan)  # 区域内保留SOC，其余NaN

def smooth_nan(arr, sigma=3):
    """NaN 安全高斯平滑：用有效像元权重归一化，避免 NaN 区域稀释"""
    valid = ~np.isnan(arr)
    filled = np.where(valid, arr, 0.0)
    num = gaussian_filter(filled, sigma=sigma, mode='nearest')
    den = gaussian_filter(valid.astype(np.float32), sigma=sigma, mode='nearest')
    den = np.where(den < 1e-6, np.nan, den)
    return num / den    

permafrost_path=r'Perma_Distr_map_TP\permafrost.tif'
permafrost=gdal.Open(permafrost_path).ReadAsArray()
seasonal_permafrost_path=r'Perma_Distr_map_TP\seasonal_permafrost.tif'
seasonal_permafrost=gdal.Open(seasonal_permafrost_path).ReadAsArray()
print('permafrost shape:', permafrost.shape,'seasonal_permafrost shape:', seasonal_permafrost.shape)
proj = gdal.Open(permafrost_path).GetProjection()
geo = gdal.Open(permafrost_path).GetGeoTransform()

SOC_folder=r'results2026/SOC image/'
depth_list=['100cm']
for i in range(2000,2021):
    for depth in depth_list:
        print(f'Processing SOC data for year {i} and depth {depth}...')
        SOC_path=SOC_folder+f'QTP_SOC_{depth}_{i}.tif'
        SOC_std_path=SOC_folder+f'QTP_SOC_{depth}_{i}_std.tif'
        SOC_img=gdal.Open(SOC_path).ReadAsArray()
        SOC_std_img=gdal.Open(SOC_std_path).ReadAsArray()
        print('SOC shape:', SOC_img.shape,'SOC_std shape:', SOC_std_img.shape)
        SOC_img_permafrost=SOC_MULTIPLY_PERMAFROST(SOC_img, permafrost)
        SOC_std_img_permafrost=SOC_MULTIPLY_PERMAFROST(SOC_std_img, permafrost)
        SOC_img_seasonal_permafrost=SOC_MULTIPLY_PERMAFROST(SOC_img, seasonal_permafrost)
        SOC_std_img_seasonal_permafrost=SOC_MULTIPLY_PERMAFROST(SOC_std_img, seasonal_permafrost)
        WriteTifImg(SOC_path, proj, geo, SOC_img, nodata=np.nan)
        
        SOC_img_permafrost=mask_soc(SOC_img, permafrost)
        SOC_img_permafrost_path=SOC_folder+f'QTP_SOC_{depth}_{i}_permafrost.tif'
        WriteTifImg(SOC_img_permafrost_path, proj, geo, SOC_img_permafrost, nodata=np.nan)
        
        SOC_img_seasonal_permafrost=mask_soc(SOC_img, seasonal_permafrost)
        SOC_img_seasonal_permafrost_path=SOC_folder+f'QTP_SOC_{depth}_{i}_seasonal_permafrost.tif'
        WriteTifImg(SOC_img_seasonal_permafrost_path, proj, geo, SOC_img_seasonal_permafrost, nodata=np.nan)

# calculate the SOCD in 0-200cm and 0-300cm for each year and save the results
SOC_folder=r'results2026/'
for i in range(2000,2021):
    print(f'Calculating SOCD for year {i}...')
    SOC_100cm_path=SOC_folder+f'QTP_SOC_100cm_{i}.tif'
    SOC_100cm_std_path=SOC_folder+f'QTP_SOC_100cm_{i}_std.tif'
    SOC_100cm_img=gdal.Open(SOC_100cm_path).ReadAsArray()
    SOC_100cm_std_img=gdal.Open(SOC_100cm_std_path).ReadAsArray()
    SOC_200cm_raster=SOC_100cm_img*1.264+0.8089
    SOC_300cm_raster=SOC_200cm_raster*1.292-0.9154
    SOC_200cm_std_raster=SOC_100cm_std_img*1.264+0.8089
    SOC_300cm_std_raster=SOC_200cm_std_raster*1.292-0.9154
    SOC_200cm_path=SOC_folder+f'QTP_SOC_200cm_{i}.tif'
    SOC_300cm_path=SOC_folder+f'QTP_SOC_300cm_{i}.tif'
    SOC_200cm_std_path=SOC_folder+f'QTP_SOC_200cm_{i}_std.tif'
    SOC_300cm_std_path=SOC_folder+f'QTP_SOC_300cm_{i}_std.tif'
    
    WriteTifImg(SOC_200cm_path, proj, geo, SOC_200cm_raster, nodata=np.nan)
    WriteTifImg(SOC_300cm_path, proj, geo, SOC_300cm_raster, nodata=np.nan)
    WriteTifImg(SOC_200cm_std_path, proj, geo, SOC_200cm_std_raster, nodata=np.nan)
    WriteTifImg(SOC_300cm_std_path, proj, geo, SOC_300cm_std_raster, nodata=np.nan)


# divide permafrost & seasonal permafrost regions and save the SOC data for each year and depth
SOC_folder=r'results2026/'
depth_list=['30cm', '50cm', '100cm','200cm', '300cm']
for i in range(2000,2021):
    for depth in depth_list:
        print(f'Processing SOC data for year {i} and depth {depth}...')
        SOC_path=SOC_folder+f'QTP_SOC_{depth}_{i}.tif'
        SOC_std_path=SOC_folder+f'QTP_SOC_{depth}_{i}_std.tif'
        SOC_img=gdal.Open(SOC_path).ReadAsArray()
        SOC_std_img=gdal.Open(SOC_std_path).ReadAsArray()
        print('SOC shape:', SOC_img.shape,'SOC_std shape:', SOC_std_img.shape)
        # SOC_img_permafrost=SOC_MULTIPLY_PERMAFROST(SOC_img, permafrost)
        # SOC_std_img_permafrost=SOC_MULTIPLY_PERMAFROST(SOC_std_img, permafrost)
        # SOC_img_seasonal_permafrost=SOC_MULTIPLY_PERMAFROST(SOC_img, seasonal_permafrost)
        # SOC_std_img_seasonal_permafrost=SOC_MULTIPLY_PERMAFROST(SOC_std_img, seasonal_permafrost)
        
        SOC_img_permafrost=mask_soc(SOC_img, permafrost)
        SOC_img_permafrost_path=SOC_folder+f'QTP_SOC_{depth}_{i}_permafrost.tif'
        WriteTifImg(SOC_img_permafrost_path, proj, geo, SOC_img_permafrost, nodata=np.nan)
                
        SOC_std_img_permafrost=mask_soc(SOC_std_img, permafrost)
        SOC_std_img_permafrost_path=SOC_folder+f'QTP_SOC_{depth}_{i}_std_permafrost.tif'
        WriteTifImg(SOC_std_img_permafrost_path, proj, geo, SOC_std_img_permafrost, nodata=np.nan)
        
        SOC_img_seasonal_permafrost=mask_soc(SOC_img, seasonal_permafrost)
        SOC_img_seasonal_permafrost_path=SOC_folder+f'QTP_SOC_{depth}_{i}_seasonal_permafrost.tif'
        WriteTifImg(SOC_img_seasonal_permafrost_path, proj, geo, SOC_img_seasonal_permafrost, nodata=np.nan)
        
        SOC_std_img_seasonal_permafrost=mask_soc(SOC_std_img, seasonal_permafrost)
        SOC_std_img_seasonal_permafrost_path=SOC_folder+f'QTP_SOC_{depth}_{i}_std_seasonal_permafrost.tif'
        WriteTifImg(SOC_std_img_seasonal_permafrost_path, proj, geo, SOC_std_img_seasonal_permafrost, nodata=np.nan)

# # calculate the average SOC for permafrost and seasonal permafrost regions for each year and depth  
# SOC_folder = r'results2026/'
SOC_folder=r'results2026/SOC image/'
depth_list = ['30cm', '50cm', '100cm','200cm', '300cm']

rows_permafrost = []
rows_seasonal_permafrost = []
for year in range(2000, 2021):
    for depth in depth_list:
        for region, suffix in [('permafrost', 'permafrost'),
                               ('seasonal', 'seasonal_permafrost')]:
            path = os.path.join(SOC_folder, f'QTP_SOC_{depth}_{year}_{suffix}.tif')
            if not os.path.exists(path):
                print('缺失文件:', path)
                continue
            arr = gdal.Open(path).ReadAsArray().astype(np.float32)
            arr = np.where(arr == -9999, np.nan, arr)      # nodata → NaN

            mean_val = np.nanmean(arr)
            std_val = np.nanstd(arr)
            n_valid = np.count_nonzero(~np.isnan(arr))

            if region == 'permafrost':
                rows_permafrost.append({'year': year, 'depth': depth, 'region': region,
                                         'mean_SOCD': mean_val,
                                         'spatial_std': std_val})
            else:
                rows_seasonal_permafrost.append({'year': year, 'depth': depth, 'region': region,
                                                  'mean_SOCD': mean_val,
                                                  'spatial_std': std_val})
            print(f'{year} {depth} {region}: mean={mean_val:.4f}')

df_permafrost = pd.DataFrame(rows_permafrost)
df_seasonal_permafrost = pd.DataFrame(rows_seasonal_permafrost)
df_permafrost.to_csv('results2026/annual_mean_SOCD_permafrost-20260901.csv', index=False, encoding='utf-8-sig')
df_seasonal_permafrost.to_csv('results2026/annual_mean_SOCD_seasonal_permafrost-20260901.csv', index=False, encoding='utf-8-sig')
print(df_permafrost)
print(df_seasonal_permafrost)

# calculate the SOC stock for permafrost and seasonal permafrost regions for each year and depth
SOC_folder = r'results2026/'
depth_list = ['30cm', '50cm', '100cm','200cm', '300cm']

def pixel_area_m2(gt, nrows, ncols):
    """WGS84 网格逐像元面积 (m²)，球面积分公式"""
    R = 6371000.0                                  # 地球平均半径 (m)
    dlam = np.deg2rad(abs(gt[1]))                  # 像元经度跨度 (弧度)
    lat_top = gt[3] - np.arange(nrows) * abs(gt[5])   # 每行上边界纬度
    lat_bot = lat_top - abs(gt[5])                    # 每行下边界纬度
    area_row = R**2 * dlam * (np.sin(np.deg2rad(lat_top)) - np.sin(np.deg2rad(lat_bot)))
    return np.broadcast_to(area_row[:, None], (nrows, ncols))

rows_permafrost = []
rows_seasonal_permafrost = []
for year in range(2000, 2021):
    for depth in depth_list:
        for region, suffix in [('permafrost', 'permafrost'),
                               ('seasonal', 'seasonal_permafrost')]:
            path = os.path.join(SOC_folder, f'QTP_SOC_{depth}_{year}_{suffix}.tif')
            if not os.path.exists(path):
                print('缺失文件:', path)
                continue
            ds = gdal.Open(path)
            arr = ds.ReadAsArray().astype(np.float32)
            arr = np.where(arr == -9999, np.nan, arr)   # nodata → NaN
            area = pixel_area_m2(ds.GetGeoTransform(), arr.shape[0], arr.shape[1])

            # 储量 = Σ(SOCD × 像元面积)；SOCD 单位为 kg C/m² 时，求和结果为 kg
            stock_Pg = np.nansum(arr * area) / 1e12
            valid_area = np.where(~np.isnan(arr), area, np.nan)
            valid_area_km2 = np.nansum(valid_area) / 1e6
            # 面积加权平均 SOCD（与简单 np.nanmean 对比用）
            aw_mean = np.nansum(arr * area) / np.nansum(valid_area)
            mean_val = np.nanmean(arr)

            # 不确定性（可选）：用 std 栅格
            stock_std_Pg = np.nan
            std_path = os.path.join(SOC_folder, f'QTP_SOC_{depth}_{year}_std_{suffix}.tif')
            if os.path.exists(std_path):
                ds_std= gdal.Open(std_path)
                std_arr = ds_std.ReadAsArray().astype(np.float32)
                std_arr = np.where(std_arr == -9999, np.nan, std_arr)
                std_area = pixel_area_m2(ds_std.GetGeoTransform(), std_arr.shape[0], std_arr.shape[1])
                stock_std_Pg = np.nansum(std_arr * std_area) / 1e12

            row = {'year': year, 'depth': depth, 'region': region,
                   'mean_SOCD_kgm2': mean_val,
                   'aw_mean_SOCD_kgm2': aw_mean,
                   'valid_area_km2': valid_area_km2,
                   'stock_Pg': stock_Pg,
                   'stock_std_Pg': stock_std_Pg}
            if region == 'permafrost':
                rows_permafrost.append(row)
            else:
                rows_seasonal_permafrost.append(row)
            print(f'{year} {depth} {region}: stock={stock_Pg:.3f} Pg, '
                  f'aw_mean={aw_mean:.4f}, mean={mean_val:.4f} kg/m2')

df_permafrost = pd.DataFrame(rows_permafrost)
df_seasonal_permafrost = pd.DataFrame(rows_seasonal_permafrost)
df_permafrost.to_csv('results2026/annual_stock_permafrost-20260914.csv', index=False, encoding='utf-8-sig')
df_seasonal_permafrost.to_csv('results2026/annual_stock_seasonal_permafrost-20260914.csv', index=False, encoding='utf-8-sig')
print(df_permafrost)
print(df_seasonal_permafrost)