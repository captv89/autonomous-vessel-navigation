"""
Simple test to verify the development environment is working correctly.
"""
import numpy as np
import matplotlib.pyplot as plt

def test_numpy():
    """Test numpy installation"""
    print("Testing NumPy...")
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"✓ NumPy version: {np.__version__}")
    print(f"✓ Created array with shape: {arr.shape}")
    return True

def test_matplotlib():
    """Test matplotlib installation"""
    print("\nTesting Matplotlib...")
    print(f"✓ Matplotlib version: {plt.matplotlib.__version__}")
    
    # Create a simple plot (won't display, just verify it works)
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    plt.close(fig)
    print("✓ Successfully created a test plot")
    return True

def main():
    print("=" * 50)
    print("Environment Setup Verification")
    print("=" * 50)
    
    try:
        test_numpy()
        test_matplotlib()
        print("\n" + "=" * 50)
        print("✓ All tests passed! Environment is ready.")
        print("=" * 50)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
