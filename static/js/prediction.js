document.addEventListener("DOMContentLoaded", function () {
    // Dynamic Live Financial & EMI Calculations on Step 2
    const applicantIncomeInput = document.getElementById("id_applicant_income");
    const coapplicantIncomeInput = document.getElementById("id_coapplicant_income");
    const loanAmountInput = document.getElementById("id_loan_amount");
    const loanTermInput = document.getElementById("id_loan_amount_term");
    const interestRateInput = document.getElementById("id_interest_rate");

    const totalIncomeDisplay = document.getElementById("totalIncomeDisplay");
    const annualIncomeDisplay = document.getElementById("annualIncomeDisplay");
    const emiDisplay = document.getElementById("emiDisplay");
    const emiRatioDisplay = document.getElementById("emiRatioDisplay");
    const healthBadge = document.getElementById("healthBadge");

    function updateLiveFinancials() {
        const appInc = parseFloat(applicantIncomeInput ? applicantIncomeInput.value : 0) || 0;
        const coAppInc = parseFloat(coapplicantIncomeInput ? coapplicantIncomeInput.value : 0) || 0;
        const rawLoanAmt = parseFloat(loanAmountInput ? loanAmountInput.value : 15) || 15;
        const tenureMonths = parseFloat(loanTermInput ? loanTermInput.value : 360) || 360;
        const annualRate = parseFloat(interestRateInput ? interestRateInput.value : 10.5) || 10.5;

        // 1. Total & Annual Income
        const monthlyIncome = appInc + coAppInc;
        const annualIncome = monthlyIncome * 12.0;

        if (totalIncomeDisplay) {
            totalIncomeDisplay.textContent = new Intl.NumberFormat('en-IN', {
                style: 'currency', currency: 'INR', maximumFractionDigits: 0
            }).format(monthlyIncome) + " / month";
        }

        if (annualIncomeDisplay) {
            annualIncomeDisplay.textContent = new Intl.NumberFormat('en-IN', {
                style: 'currency', currency: 'INR', maximumFractionDigits: 0
            }).format(annualIncome) + " / year";
        }

        // 2. Loan Principal in Rupees
        let loanInr = rawLoanAmt;
        if (rawLoanAmt < 10000.0) {
            loanInr = rawLoanAmt * 100000.0; // Lakhs to Rupees
        }

        // 3. Bank Standard EMI Formula
        const r = annualRate / (12.0 * 100.0);
        const n = tenureMonths;
        let emi = 0;
        if (n > 0 && r > 0) {
            emi = loanInr * r * (Math.pow(1 + r, n)) / (Math.pow(1 + r, n) - 1);
        } else {
            emi = loanInr / (n > 0 ? n : 1);
        }

        if (emiDisplay) {
            emiDisplay.textContent = new Intl.NumberFormat('en-IN', {
                style: 'currency', currency: 'INR', maximumFractionDigits: 0
            }).format(emi) + " / month";
        }

        // 4. EMI to Income Burden Ratio & Health Badge
        let emiRatio = 0;
        if (monthlyIncome > 0) {
            emiRatio = (emi / monthlyIncome) * 100.0;
        }

        if (emiRatioDisplay) {
            emiRatioDisplay.textContent = emiRatio.toFixed(1) + "%";
        }

        if (healthBadge) {
            if (emiRatio <= 30.0) {
                healthBadge.className = "badge bg-success px-3 py-2 fs-6";
                healthBadge.innerHTML = '<i class="fa-solid fa-circle-check me-1"></i> Healthy Repayment Burden (≤ 30%)';
            } else if (emiRatio <= 40.0) {
                healthBadge.className = "badge bg-warning text-dark px-3 py-2 fs-6";
                healthBadge.innerHTML = '<i class="fa-solid fa-triangle-exclamation me-1"></i> Moderate Repayment Burden (30% - 40%)';
            } else {
                healthBadge.className = "badge bg-danger px-3 py-2 fs-6";
                healthBadge.innerHTML = '<i class="fa-solid fa-triangle-exclamation me-1"></i> ⚠️ High Repayment Burden (> 40%)';
            }
        }
    }

    if (applicantIncomeInput) applicantIncomeInput.addEventListener("input", updateLiveFinancials);
    if (coapplicantIncomeInput) coapplicantIncomeInput.addEventListener("input", updateLiveFinancials);
    if (loanAmountInput) loanAmountInput.addEventListener("input", updateLiveFinancials);
    if (loanTermInput) loanTermInput.addEventListener("input", updateLiveFinancials);
    if (interestRateInput) interestRateInput.addEventListener("input", updateLiveFinancials);

    updateLiveFinancials();

    // Multi-Step Form Navigation
    const step1 = document.getElementById("step1");
    const step2 = document.getElementById("step2");
    const step3 = document.getElementById("step3");

    const btnNext1 = document.getElementById("btnNext1");
    const btnPrev2 = document.getElementById("btnPrev2");
    const btnNext2 = document.getElementById("btnNext2");
    const btnPrev3 = document.getElementById("btnPrev3");

    const node1 = document.getElementById("node1");
    const node2 = document.getElementById("node2");
    const node3 = document.getElementById("node3");

    if (btnNext1 && btnNext2) {
        btnNext1.addEventListener("click", function () {
            step1.style.display = "none";
            step2.style.display = "block";
            node1.classList.remove("active");
            node1.classList.add("completed");
            node2.classList.add("active");
        });

        btnPrev2.addEventListener("click", function () {
            step2.style.display = "none";
            step1.style.display = "block";
            node2.classList.remove("active");
            node1.classList.remove("completed");
            node1.classList.add("active");
        });

        btnNext2.addEventListener("click", function () {
            step2.style.display = "none";
            step3.style.display = "block";
            node2.classList.remove("active");
            node2.classList.add("completed");
            node3.classList.add("active");
        });

        btnPrev3.addEventListener("click", function () {
            step3.style.display = "none";
            step2.style.display = "block";
            node3.classList.remove("active");
            node2.classList.remove("completed");
            node2.classList.add("active");
        });
    }

    // Form Submission Loading Animation
    const loanForm = document.getElementById("loanForm");
    const loadingOverlay = document.getElementById("loadingOverlay");

    if (loanForm && loadingOverlay) {
        loanForm.addEventListener("submit", function (e) {
            loadingOverlay.style.display = "flex";
            
            // Updated animation sequence without SHAP
            setTimeout(() => {
                const s1 = document.getElementById("status1");
                if (s1) s1.innerHTML = '<span class="text-success"><i class="fa-solid fa-check me-2"></i>Validating applicant information...</span>';
            }, 300);

            setTimeout(() => {
                const s2 = document.getElementById("status2");
                if (s2) s2.innerHTML = '<span class="text-success"><i class="fa-solid fa-check me-2"></i>Calculating financial indicators & EMI...</span>';
            }, 700);

            setTimeout(() => {
                const s3 = document.getElementById("status3");
                if (s3) s3.innerHTML = '<span class="text-success"><i class="fa-solid fa-check me-2"></i>Preparing model features & preprocessing...</span>';
            }, 1100);

            setTimeout(() => {
                const s4 = document.getElementById("status4");
                if (s4) s4.innerHTML = '<span class="text-success"><i class="fa-solid fa-check me-2"></i>Running ML models & Neural Network...</span>';
            }, 1600);

            setTimeout(() => {
                const s5 = document.getElementById("status5");
                if (s5) s5.innerHTML = '<span class="text-success"><i class="fa-solid fa-check me-2"></i>Comparing model performance & risk score...</span>';
            }, 2000);
        });
    }
});
