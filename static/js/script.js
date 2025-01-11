document.addEventListener("DOMContentLoaded", function () {
    const startBtn = document.getElementById("startBot");
    const stopBtn = document.getElementById("stopBot");
    const saveSettingsBtn = document.getElementById("saveSettingsButton");
    const resetBtn = document.getElementById("resetBot");

    function showFeedback(message, type) {
        const feedbackEl = document.getElementById("feedback");
        if (!feedbackEl) return;
        feedbackEl.textContent = message || "An unexpected error occurred.";
        feedbackEl.className = `alert alert-${type}`;
        feedbackEl.style.display = "block";
        setTimeout(() => {
            feedbackEl.style.display = "none";
        }, 5000);
    }    

    function handleBotControl(endpoint, successMessage, errorMessage) {
        fetch(endpoint, { method: "POST" })
            .then(response => response.json())
            .then(data => {
                const type = data.success ? "success" : "danger";
                const message = data.message || (data.success ? successMessage : errorMessage);
                showFeedback(message, type);
            })
            .catch(() => {
                showFeedback(errorMessage, "danger");
            });
    }     
    
    if (startBtn) {
        startBtn.addEventListener("click", () =>
            handleBotControl(
                "/start_bot",
                "Bot started successfully!",
                "Failed to start bot."
            )
        );
    }
    
    if (stopBtn) {
        stopBtn.addEventListener("click", () =>
            handleBotControl(
                "/stop_bot",
                "Bot stopped successfully!",
                "Failed to stop bot."
            )
        );
    }
    
    if (resetBtn) {
        resetBtn.addEventListener("click", () =>
            handleBotControl(
                "/reset_bot",
                "Bot reset successfully!",
                "Failed to reset bot."
            )
        );
    }            

    function parseTradeInterval(value) {
        if (value.endsWith('m')) {
            return parseFloat(value) / 60;
        } else if (value.endsWith('h')) {
            return parseFloat(value);
        }
        return 1;
    }

    function loadSettings() {
        fetch("/settings", { method: "GET" })
            .then(response => {
                if (!response.ok) throw new Error("Failed to load settings");
                return response.json();
            })
            .then(settings => {
                console.log("Loaded Settings:", settings);

                document.getElementById("riskLevel").value = settings.riskLevel || "medium";
                document.getElementById("maxTrades").value = settings.maxTrades || 5;
                document.getElementById("tradeInterval").value =
                    settings.tradeIntervalHours === 1 ? "1h" : "1m";
                document.getElementById("stopLossPercentage").value =
                    settings.stopLossPercentage !== undefined ? settings.stopLossPercentage : 2.0;
                document.getElementById("takeProfitPercentage").value =
                    settings.takeProfitPercentage !== undefined ? settings.takeProfitPercentage : 5.0;
                document.getElementById("rsiPeriod").value =
                    settings.rsiPeriod !== undefined ? settings.rsiPeriod : 14;
                document.getElementById("emaFastPeriod").value =
                    settings.emaFastPeriod !== undefined ? settings.emaFastPeriod : 12;
                document.getElementById("emaSlowPeriod").value =
                    settings.emaSlowPeriod !== undefined ? settings.emaSlowPeriod : 26;
                document.getElementById("bbandsPeriod").value =
                    settings.bbandsPeriod !== undefined ? settings.bbandsPeriod : 20;
                document.getElementById("trailingATRMultiplier").value =
                    settings.trailingATRMultiplier !== undefined ? settings.trailingATRMultiplier : 1.5;
                document.getElementById("adjustATRMultiplier").value =
                    settings.adjustATRMultiplier !== undefined ? settings.adjustATRMultiplier : 1.5;
                document.getElementById("tradeCooldown").value =
                    settings.tradeCooldown !== undefined ? settings.tradeCooldown : 60;
            })
            .catch(error => {
                console.error("Error loading settings:", error);
                showFeedback("Failed to load settings.", "danger");
            });
    }

    function saveSettings() {
        const settings = {
            riskLevel: document.getElementById("riskLevel").value,
            maxTrades: parseInt(document.getElementById("maxTrades").value, 10),
            tradeIntervalHours: parseTradeInterval(document.getElementById("tradeInterval").value),
            stopLossPercentage: parseFloat(document.getElementById("stopLossPercentage").value),
            takeProfitPercentage: parseFloat(document.getElementById("takeProfitPercentage").value),
            rsiPeriod: parseInt(document.getElementById("rsiPeriod").value, 10),
            emaFastPeriod: parseInt(document.getElementById("emaFastPeriod").value, 10),
            emaSlowPeriod: parseInt(document.getElementById("emaSlowPeriod").value, 10),
            bbandsPeriod: parseInt(document.getElementById("bbandsPeriod").value, 10),
            trailingATRMultiplier: parseFloat(document.getElementById("trailingATRMultiplier").value),
            adjustATRMultiplier: parseFloat(document.getElementById("adjustATRMultiplier").value),
            tradeCooldown: parseInt(document.getElementById("tradeCooldown").value, 10),
        };

        if (isNaN(settings.maxTrades) || isNaN(settings.tradeCooldown)) {
            showFeedback("Please fill out all fields with valid data.", "warning");
            return;
        }

        fetch("/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(settings),
        })
            .then(response => {
                if (!response.ok) throw new Error("Failed to save settings");
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    showFeedback(data.error, "danger");
                } else if (data.message) {
                    showFeedback(data.message, "success");
                } else {
                    showFeedback("Unknown response from server.", "warning");
                }
            })
            .catch(error => {
                console.error("Error saving settings:", error);
                showFeedback("Failed to save settings.", "danger");
            });
    }

    if (saveSettingsBtn) {
        saveSettingsBtn.addEventListener("click", saveSettings);
    }

    function fetchOpenPositions() {
        const tableBody = document.getElementById("openPositionsTable");
        if (!tableBody) {
            console.warn("[fetchOpenPositions] Table body not found.");
            return;
        }

        fetch("/open_positions")
            .then(res => {
                if (!res.ok) throw new Error(`HTTP error! Status: ${res.status}`);
                return res.json();
            })
            .then(positions => {
                tableBody.innerHTML = "";
                if (Array.isArray(positions) && positions.length > 0) {
                    positions.forEach(pos => {
                        tableBody.innerHTML += `
                            <tr>
                                <td>${pos.instrument || "N/A"}</td>
                                <td>${pos.side}</td>
                                <td>${pos.units}</td>
                                <td>${pos.entryPrice !== undefined ? pos.entryPrice.toFixed(5) : "N/A"}</td>
                                <td>${pos.currentPrice !== undefined ? pos.currentPrice.toFixed(5) : "N/A"}</td>
                                <td>${pos.profitLoss !== undefined ? pos.profitLoss.toFixed(2) : "N/A"}</td>
                            </tr>
                        `;
                    });
                } else {
                    tableBody.innerHTML = `
                        <tr>
                            <td colspan="6" class="text-center">No open positions available</td>
                        </tr>`;
                }
            })
            .catch(error => {
                console.error("[fetchOpenPositions] error:", error);
                showFeedback("Failed to fetch open positions.", "danger");
            });
    }

    function fetchTradeHistory() {
        const tableBody = document.getElementById("tradeHistoryTable");
        if (!tableBody) {
            console.warn("[fetchTradeHistory] Table body not found.");
            return;
        }

        fetch("/history_data")
            .then(res => {
                if (!res.ok) throw new Error(`HTTP error! Status: ${res.status}`);
                return res.json();
            })
            .then(data => {
                tableBody.innerHTML = "";
                if (Array.isArray(data) && data.length > 0) {
                    data.forEach(trade => {
                        const exitPx = trade.exitPrice !== undefined ? trade.exitPrice.toFixed(5) : "N/A";
                        const profitLoss = trade.profitLoss !== undefined ? trade.profitLoss.toFixed(2) : "N/A";
                        tableBody.innerHTML += `
                            <tr>
                                <td>${trade.time || "N/A"}</td>
                                <td>${trade.instrument || "N/A"}</td>
                                <td>${trade.side || "N/A"}</td>
                                <td>${trade.units || "N/A"}</td>
                                <td>${trade.entryPrice !== undefined ? trade.entryPrice.toFixed(5) : "N/A"}</td>
                                <td>${exitPx}</td>
                                <td>${profitLoss}</td>
                            </tr>`;
                    });
                } else {
                    tableBody.innerHTML = `
                        <tr>
                            <td colspan="7" class="text-center">No trade history available</td>
                        </tr>`;
                }
            })
            .catch(error => {
                console.error("[fetchTradeHistory] error:", error);
                showFeedback("Failed to fetch trade history.", "danger");
            });
    }

    function initializeMetricsSSE() {
        const source = new EventSource("/metrics_stream"); 
        source.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                const tradeCountEl = document.getElementById("tradeCount");
                const profitLossEl = document.getElementById("profitLoss");
                const accountBalanceEl = document.getElementById("accountBalance");
                const timeElapsedEl = document.getElementById("timeElapsed");

                if (tradeCountEl) {
                    tradeCountEl.innerText = data.tradeCount || 0;
                }
                if (profitLossEl) {
                    profitLossEl.innerText = `$${parseFloat(data.profitLoss || 0).toFixed(2)}`;
                }
                if (accountBalanceEl) {
                    accountBalanceEl.innerText = `$${parseFloat(data.accountBalance || 0).toFixed(2)}`;
                }
                if (timeElapsedEl) {
                    timeElapsedEl.innerText = data.timeElapsed || "00:00:00";
                }
            } catch (error) {
                console.error("SSE parse error:", error);
            }
        };

        source.onerror = (err) => {
            console.error("SSE connection error:", err);
            showFeedback("Lost connection to metrics. Reconnecting...", "warning");
            source.close();
            setTimeout(() => initializeMetricsSSE(), 5000);
        };
    }

    const currentPath = window.location.pathname;
    if (currentPath === "/overview") {
        initializeMetricsSSE();
    } else if (currentPath === "/positions") {
        fetchOpenPositions();
    } else if (currentPath === "/history") {
        fetchTradeHistory();
    } else if (currentPath === "/settings") { // Corrected from "/settings_page"
        loadSettings();
    }
});
