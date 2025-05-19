# Helper functions for PCA written for Unit 1 Summative Program. Changes made from grading feedback
# I like to add comments and spacing so my code is a bit long but easy to read
import numpy as np

def main():
    C = [[1, 2],
         [3, 4]]
    print(createPCA(C, 2))


def multiply(A, B):
    # Step 1: Find Dimensions of A and B
    aMSize = len(A)
    aNSize = len(A[0])
    bMSize = len(B)
    bNSize = len(B[0])

    # Step 2: Confirm A and B are compatible
    if aNSize != bMSize:
        print("Matrices are not compatible for multiplication! Please try again.")
    
    # Step 3: Create empty array
    product_matrix = np.array([[0.0 for i in range(bNSize)] for i in range(aMSize)])

    # Step 4: 3x Nested for Loop to multiply each element -- each row of A > each column of B > each column of A
    for i in range(aMSize):
        for j in range(bNSize):
            for k in range(aNSize):
              product_matrix[i][j] += A[i][k] * B[k][j]    # multiplying and adding in one line

    # Step 5: Return
    return product_matrix


def transpose(A):
    # Step 1: Find Dimension of A
    aMSize = len(A)
    aNSize = len(A[0])

    # Step 2: Create empty array
    transpose_matrix = np.array([[0.0 for i in range(aMSize)] for i in range(aNSize)])

    # Step 3: 2x nested for loop to transpose each element
    for i in range(aNSize):
        for j in range(aMSize):
            transpose_matrix[i][j] = A[j][i]
    
    # Step 4: Return
    return transpose_matrix


def calcEigenvalues(A):
    '''
    Calculates all of the eigenvalues of a matrix.
    param A: matrix of dimension m by n
    return: list of eigenvalues in descending order
    '''
    eigenvalues, Eigenvectors = np.linalg.eig(A)
    return np.sort(eigenvalues)[::-1]


def calcEigenvectors(A):
    '''
    Calculates all of the eigenvalues of a matrix
    param A: matrix of dimension m by n
    return: orthogonal matrix of eigenvalues
    '''
    eigenvalues, eigenvectors = np.linalg.eig(A)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    sorted_eigenvectors = []
    # Get index values for eigenvectors from unsorted eigenvalues vector
    for num in sorted_eigenvalues:
        index = np.argmax(eigenvalues == num)
        sorted_eigenvectors.append(transpose(eigenvectors)[index])
    return np.array(sorted_eigenvectors)


def calcSigma(A, eigenvalues):
    # Step 1: Find Dimension of A
    aMSize = len(A)
    aNSize = len(A[0])

    # Step 2: Create empty array
    # Fixed per Riley feedback
    sigma_matrix = np.array([[0.0 for i in range(aNSize)] for i in range(aMSize)])

    # Step 3: Create Sigmas list
    sigmas = []

    # Step 4: Loop over eigenvalues and square root them
    for i in eigenvalues:
        sigmas.append(i** 0.5)

    # Step 5: Fill sigma matrix 
    for i in range(min(aMSize, aNSize, len(sigmas))):
        sigma_matrix[i][i] = float(sigmas[i])
    
    return sigma_matrix

# Helper Function for U which is A * Atransposed
def calcUMatrix(A):
    ATranspose = transpose(A)
    ATimesATransposed = multiply(A, ATranspose)
    UMatrix = transpose(calcEigenvectors(ATimesATransposed))
    return UMatrix

# Helper Function for V which is Atransposed * A
def calcVMatrix(A):
    ATranspose = transpose(A)
    ATransposedTimesA = multiply(ATranspose, A)
    VMatrix = calcEigenvectors(ATransposedTimesA)
    return VMatrix


def createSVD(A):
    # Step 1: Calculate U Matrix
    UofA = np.array(calcUMatrix(A))

    # Step 2: Calculate Sigma
    ATransposed = transpose(A)
    ATransposedTimesA = multiply(ATransposed, A)
    eigenvalues = calcEigenvalues(ATransposedTimesA)
    Sigma = np.array(calcSigma(A, eigenvalues))

    #Step 3: Calculate V Transposed
    VofA = calcVMatrix(A)
    VTranspose = np.array(transpose(VofA))

    # Step 4: Return Matrices
    return np.array(UofA), np.array(Sigma), np.array(VTranspose)  
    
# Helper function for calculating row mean
def calcRowMean(A):
    # Create list of row means and placeholder variable
    rowMeanList = []
    # Loop over each row to get the mean
    for i in range(len(A)):
        rowMean = 0
        for j in range(len(A[0])):
            rowMean += A[i][j]
        rowMean /= len(A[0])
        # Append each rowMean to the RowMeanList
        rowMeanList.append(rowMean)
    # Return RowMeanList
    return rowMeanList

# Helper function to center data
def centerData(A):
    # Calculate Row Mean of A
    aRowMean = calcRowMean(A)

    # Create lists for centered data
    centeredDataList = []

    # Loop through each element and row to get means
    for i in range(len(A)):
        tempCenteredList = []
        for j in range(len(A[0])):
            tempCenteredList.append(A[i][j] - aRowMean[i])
        centeredDataList.append(tempCenteredList)
    
    # Return centeredDataList
    return centeredDataList

# Helper function to calculate covariance
def calcCovariance(centeredDataList):
    # Get size of centeredDataList
    aMSize = len(centeredDataList)
    aNSize = len(centeredDataList[0])

    # Create covariance matrix to fill
    covarianceMatrix = np.array([[0.0 for i in range(aMSize)] for i in range(aMSize)])

    # 3x for loop - similar to multiply function 
    for i in range(aMSize):
        for j in range(aMSize):
            for k in range(aNSize):
                covarianceMatrix[i][j] += (centeredDataList[i][k] * centeredDataList[j][k]) / len(centeredDataList)
    
    # Return covarianceMatrix
    return covarianceMatrix

# Helper function to calculate principle components
def calcPrincipleComponents(U, components):
    # Create principle components
    principleComponents = []

    # 2x for loop - append principle components to temp list
    for i in range((len(U))):
        tempPCs = []
        for j in range(components):
            tempPCs.append(U[i][j])
        principleComponents.append(tempPCs)
    
    # Return principleComponents
    return principleComponents


def createPCA(A, components):
    # Put it all together
    centeredA = centerData(A)
    covarianceOfA = calcCovariance(centeredA)
    U, Sigma, VTransposed = createSVD(covarianceOfA)
    principleComponentsOfA = calcPrincipleComponents(U, components)

    # Return PCA
    return np.array(principleComponentsOfA)


if __name__ == "__main__":
    main()