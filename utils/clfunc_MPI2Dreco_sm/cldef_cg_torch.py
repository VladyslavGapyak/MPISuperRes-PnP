import torch

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearOperator_torch:
    def __init__(self,size,matvec):
        self.size = size
        self.matvec = matvec

    def matvec(self,b):
        '''
        Method to call the Sparse Linear Matrix Application
        '''
        if b.size()!=self.size[1]:
            raise Exception("The vector b does not match the size of the linear Operator for matrix multiplication!")
        
        return self.matvec(b)
    
    def cg(self,b,x0=None,maxit=1000,tol=1e-5):
        '''
        CG method with starting point x0, solving the liner system of equations Ax=b
        How to use: if A = LinearOperator_torch(size=some_size,matvec=some_function)
        then result = A.cg(b,x=,maxit,tol).

        Inputs:
        
            b : the right hand side of the linear system
            x0 : initial guess if available, if None, it will be initalized to zero
            maxit : maximum number of iterations
            tol : residula tolerance for convergence

        Stopping Criterion:
            the method stops if either maxit is reached or residual is < tol.

        Output variables: x, converged, i 
            x : final solution to Ax=b
            converged : 0 if tolerance reached before maxit
                        1 otherwise
            i : number of iterations needed for stopping 
        '''
        if x0==None:
            x0 = torch.zeros_like(b)

        if b.size() != x0.size():
            raise Exception("The initial point and the vector b have not the same size!")
        
        norm_b = torch.norm(b)
        
        x = x0
        r = b - self.matvec(x)
        p = r
        rsold = torch.dot(r,r)

        converged = 1

        for i in range(maxit):
            Ap = self.matvec(p)
            alpha = rsold / torch.dot(p,Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = torch.dot(r,r)
            sqrt_res = torch.sqrt(rsnew)
            if sqrt_res < tol*norm_b:
                print("Tollerance reached!")
                converged = 0
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew

            #print(f"CG iteration {i+1}/{maxit}, [RES] : {sqrt_res}")

        print(f"[Converged] : {converged} , [IT] : {i}")

        return x,converged,i


    
    if __name__ == "__main__":

        from cldef_cg_torch import LinearOperator_torch

        b = torch.tensor([1,2],device = dev)

        def functA(vec):
            matrix = torch.tensor([[4,1],
                                   [1,3]], device =dev,dtype=torch.float64)
            
            return torch.mv(matrix,vec)

        A = LinearOperator_torch((b.size(),b.size()), matvec = functA)

        x0 = torch.tensor([2,1],device=dev,dtype=torch.float64)

        x,converged,i = A.cg(b,x0=x0,maxit=10000,toll=1e-16)

        print(x)



