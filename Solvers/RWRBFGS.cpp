
#include "Solvers/RWRBFGS.h"

/*Define the namespace*/
namespace ROPTLIB{

	RWRBFGS::RWRBFGS(const Problem *prob, const Variable *initialx, LinearOPE *initialH)
	{
		Initialization(prob, initialx, initialH);
	};

	void RWRBFGS::Initialization(const Problem *prob, const Variable *initialx, LinearOPE *initialH)
	{
        SetDefaultParams();
		SetProbX(prob, initialx, initialH);
	};

	void RWRBFGS::SetProbX(const Problem *prob, const Variable *initialx, LinearOPE *initialH)
	{
		SolversSMLS::SetProbX(prob, initialx);
        bool initHisnull = (initialH == nullptr);
        if (initHisnull)
        {
            if (prob->GetDomain()->GetIsIntrinsic())
            {
                initialH = new LinearOPE(prob->GetDomain()->GetEMPTYINTR().Getlength(), prob->GetDomain()->GetEMPTYINTR().Getlength());
            }
            else
            {
                initialH = new LinearOPE(prob->GetDomain()->GetEMPTYEXTR().Getlength(), prob->GetDomain()->GetEMPTYEXTR().Getlength());
            }
            initialH->ScaledIdOPE();
        }
        H = *initialH;
        if (initHisnull)
            delete initialH;
        prob->SetUseGrad(true);
        prob->SetUseHess(false);
        s = Prob->GetDomain()->GetEMPTY();
        y = Prob->GetDomain()->GetEMPTY();
	};

	void RWRBFGS::SetDefaultParams()
	{
		SolversSMLS::SetDefaultParams();
		isconvex = false;
		nu = static_cast<realdp> (1e-4);
		mu = 1;
		InitSteptype = LSSM_QUADINTMOD;
		SolverName.assign("RWRBFGS");
	};

    void RWRBFGS::SetParams(PARAMSMAP params)
    {
        SolversSMLS::SetParams(params);
        PARAMSMAP::iterator iter;
        for (iter = params.begin(); iter != params.end(); iter++)
        {
            if (iter->first == static_cast<std::string> ("isconvex"))
            {
                isconvex = ((static_cast<integer> (iter->second)) != 0);
            }
            else
            if (iter->first == static_cast<std::string> ("nu"))
            {
                nu = iter->second;
            }
            else
            if (iter->first == static_cast<std::string> ("mu"))
            {
                mu = iter->second;
            }
        }
    };

	RWRBFGS::~RWRBFGS(void)
	{
	};

	void RWRBFGS::CheckParams(void)
	{
		SolversSMLS::CheckParams();
		char YES[] = "YES";
		char NO[] = "NO";
		char *status;

		printf("RWRBFGS METHOD PARAMETERS:\n");
		status = (nu >= 0 && nu < 1) ? YES : NO;
		printf("nu            :%15g[%s],\t", nu, status);
		status = (mu >= 0) ? YES : NO;
		printf("mu            :%15g[%s],\n", mu, status);
		status = YES;
		printf("isconvex      :%15d[%s],\n", isconvex, status);
	};

	void RWRBFGS::GetSearchDir(void)
	{
		HvRWRBFGS(gf1, H, &eta1);
		Mani->ScalarTimesVector(x1, -1.0, eta1, &eta1);
	};

	void RWRBFGS::UpdateData(void)
	{
		UpdateDataRWRBFGS();
	};

	void RWRBFGS::PrintInfo(void)
	{
        printf("i:%d,f:%.3e,df/f:%.3e,", iter, f2, ((f1 - f2) / std::fabs(f2)));

        printf("|gf|:%.3e,t0:%.2e,t:%.2e,s0:%.2e,s:%.2e,time:%.2g,", ngf2, initiallength, stepsize, initialslope, newslope, static_cast<realdp>(getTickCount() - starttime) / CLK_PS);

        printf("\n\tinpss:%.3e,inpsy:%.3e,IsUpdateHessian:%d,", inpss, inpsy, isupdated);
        
        printf("nf:%d,ng:%d,", nf, ng);
        
        if (nH != 0)
            printf("nH:%d,", nH);
        
        printf("nR:%d,", nR);
        
        if (nV != 0)
            printf("nV(nVp):%d(%d),", nV, nVp);
        
        printf("\n");
	};

    Vector &RWRBFGS::HvRWRBFGS(const Vector &v, const LinearOPE &H, Vector *result)
    {
        nH++;
        return Mani->LinearOPEEta(x1, H, v, result);
    };

    void RWRBFGS::UpdateDataRWRBFGS()
    {
        s = eta2;
        Mani->coTangentVector(x1, eta2, x2, gf2, &y);
        Mani->VectorLinearCombination(x1, -1, gf1, 1, y, &y); nV++;
        inpsy = Mani->Metric(x1, s, y);
        
        if (isconvex && iter == 1 && inpsy > 0)
            H.ScaledIdOPE(inpsy / Mani->Metric(x1, y, y));
        
        inpss = Mani->Metric(x1, s, s);
        if (inpsy / inpss >= nu * pow(ngf2, mu) && (ngf2 / ngf0 < 1e-3 ||
        (inpss > std::numeric_limits<realdp>::epsilon() && inpsy > std::numeric_limits<realdp>::epsilon())))
        {
            Vector Hy(y); Mani->LinearOPEEta(x2, H, y, &Hy); nH++;
            Mani->HaddScaledRank1OPE(x2, H, static_cast<realdp> (-1) / inpsy, s, Hy, &H);
            Mani->LinearOPEEta(x2, H, y, &Hy);
            Mani->HaddScaledRank1OPE(x2, H, static_cast<realdp> (-1) / inpsy, Hy, s, &H);
            Mani->HaddScaledRank1OPE(x2, H, static_cast<realdp> (1) / inpsy, s, s, &H);
            Mani->TranHInvTran(x1, eta2, x2, H, &H);
            isupdated = true;
        }
        else
        {
            isupdated = false;
            Mani->TranHInvTran(x1, eta2, x2, H, &H);
        }
    };
}; /*end of ROPTLIB namespace*/
