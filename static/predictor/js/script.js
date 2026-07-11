/**
 * Loan Prediction System - Main JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize form validation
    initializeFormValidation();
    
    // Auto-dismiss alerts
    autoDismissAlerts();
    
    // Add loading state to forms
    addFormLoadingState();
});

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize form validation for prediction form
 */
function initializeFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
    
    // Real-time validation for numeric inputs
    const numericInputs = document.querySelectorAll('input[type="number"]');
    numericInputs.forEach(input => {
        input.addEventListener('input', function() {
            if (this.value && parseFloat(this.value) < 0) {
                this.classList.add('is-invalid');
                this.setCustomValidity('Value cannot be negative');
            } else {
                this.classList.remove('is-invalid');
                this.setCustomValidity('');
            }
        });
    });
}

/**
 * Auto-dismiss alerts after 5 seconds
 */
function autoDismissAlerts() {
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    alerts.forEach(alert => {
        setTimeout(() => {
            const closeButton = alert.querySelector('.btn-close');
            if (closeButton) {
                closeButton.click();
            }
        }, 5000);
    });
}

/**
 * Add loading state to form submissions
 */
function addFormLoadingState() {
    const forms = document.querySelectorAll('form[data-loading]');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                const originalText = submitBtn.innerHTML;
                submitBtn.disabled = true;
                submitBtn.innerHTML = `
                    <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                    Processing...
                `;
                
                // Store original text for restoration
                submitBtn.dataset.originalText = originalText;
            }
        });
    });
}

/**
 * Format currency
 */
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(amount);
}

/**
 * Format percentage
 */
function formatPercentage(value) {
    return value.toFixed(1) + '%';
}

/**
 * Get color based on prediction
 */
function getPredictionColor(prediction) {
    return prediction === 'Approved' ? 'success' : 'danger';
}

/**
 * Get icon based on prediction
 */
function getPredictionIcon(prediction) {
    return prediction === 'Approved' ? 'fa-check-circle' : 'fa-times-circle';
}

/**
 * Copy text to clipboard
 */
function copyToClipboard(text) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
            showToast('Copied to clipboard!', 'success');
        }).catch(() => {
            fallbackCopy(text);
        });
    } else {
        fallbackCopy(text);
    }
}

/**
 * Fallback copy method
 */
function fallbackCopy(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    try {
        document.execCommand('copy');
        showToast('Copied to clipboard!', 'success');
    } catch (err) {
        showToast('Failed to copy', 'danger');
    }
    document.body.removeChild(textArea);
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    const toastContainer = document.querySelector('.toast-container') || createToastContainer();
    
    const toastEl = document.createElement('div');
    toastEl.className = `toast align-items-center text-white bg-${type} border-0`;
    toastEl.setAttribute('role', 'alert');
    toastEl.setAttribute('aria-live', 'assertive');
    toastEl.setAttribute('aria-atomic', 'true');
    
    toastEl.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    toastContainer.appendChild(toastEl);
    const toast = new bootstrap.Toast(toastEl, { delay: 3000 });
    toast.show();
    
    toastEl.addEventListener('hidden.bs.toast', function() {
        this.remove();
    });
}

/**
 * Create toast container
 */
function createToastContainer() {
    const container = document.createElement('div');
    container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
    container.style.zIndex = '1050';
    document.body.appendChild(container);
    return container;
}

/**
 * Validate credit history
 */
function validateCreditHistory(value) {
    const num = parseFloat(value);
    return !isNaN(num) && (num === 0 || num === 1);
}

/**
 * Validate income fields
 */
function validateIncome(value) {
    const num = parseFloat(value);
    return !isNaN(num) && num >= 0;
}

/**
 * Get query parameter from URL
 */
function getQueryParam(param) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(param);
}

/**
 * Scroll to element smoothly
 */
function scrollToElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Export functions for global use
window.formatCurrency = formatCurrency;
window.formatPercentage = formatPercentage;
window.getPredictionColor = getPredictionColor;
window.getPredictionIcon = getPredictionIcon;
window.copyToClipboard = copyToClipboard;
window.showToast = showToast;
window.validateCreditHistory = validateCreditHistory;
window.validateIncome = validateIncome;
window.scrollToElement = scrollToElement;