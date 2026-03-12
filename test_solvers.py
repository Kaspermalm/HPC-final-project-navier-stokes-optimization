import pytest
import h5py
import navier_stokes_spectral_refactored
import nv_gpu_optimized
import nv_refactored_gpu_optimized
import navier_stokes_spectral_pyfftw_algo_optim_vortex_planned_real
import navier_stokes_spectral_pyfftw_algo_optim
import numpy as np

@pytest.fixture(autouse=True)
def setup_and_teardown():
    print("\nFetching test data")
    yield
    print("\nTest run finished")

@pytest.fixture(scope='session')
def get_validation_data():
    with h5py.File('benchmark_data.hdf5', 'r') as f:
        wz = f['/true_data/wz'][:]
        vx = f['/true_data/vx'][:]
        vy = f['/true_data/vy'][:]
    return wz, vx, vy

def test_refactored(get_validation_data):
    wz_true, vx_true, vy_true = get_validation_data
    wz, vx, vy = navier_stokes_spectral_refactored.main()
    assert np.allclose(wz, wz_true,rtol=10**-4)
    assert np.allclose(vx, vx_true,rtol=10**-4)
    assert np.allclose(vy, vy_true,rtol=10**-4)


def test_pyfftw(get_validation_data):
    wz_true, vx_true, vy_true = get_validation_data
    wz, vx, vy = navier_stokes_spectral_pyfftw_algo_optim.main()
    assert np.allclose(wz, wz_true,rtol=10**-4)
    assert np.allclose(vx, vx_true,rtol=10**-4)
    assert np.allclose(vy, vy_true,rtol=10**-4)

def test_pyfftw_refactored(get_validation_data):
    wz_true, vx_true, vy_true = get_validation_data
    wz, vx, vy = navier_stokes_spectral_pyfftw_algo_optim_vortex_planned_real.main()
    assert np.allclose(wz, wz_true,rtol=10**-4)
    assert np.allclose(vx, vx_true,rtol=10**-4)
    assert np.allclose(vy, vy_true,rtol=10**-4)

def test_gpu(get_validation_data):
    wz_true, vx_true, vy_true = get_validation_data
    wz, vx, vy = nv_gpu_optimized.main()
    assert np.allclose(wz, wz_true,rtol=10**-4)
    assert np.allclose(vx, vx_true,rtol=10**-4)
    assert np.allclose(vy, vy_true,rtol=10**-4)

def test_gpu_refactored(get_validation_data):
    wz_true, vx_true, vy_true = get_validation_data
    wz, vx, vy = nv_refactored_gpu_optimized.main()
    assert np.allclose(wz, wz_true,rtol=10**-4)
    assert np.allclose(vx, vx_true,rtol=10**-4)
    assert np.allclose(vy, vy_true,rtol=10**-4)
