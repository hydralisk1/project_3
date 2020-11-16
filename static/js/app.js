function chart(){
    var chart_stock = stock_prices.slice(-30);

    var trace1 = {
        x: chart_stock.map(d => d.date),
        y: chart_stock.map(d => d.close),
        mode: "lines",
        name: "Actual Prices",
        type: "scatter"
    };

    var next_days = d => {
        var days = [];
        var last_day = new Date(d);
        var format = d3.timeFormat("%Y-%m-%d");
        days.push(format(last_day.setDate(last_day.getDate()+1)));
        format(last_day.setDate(last_day.getDate()+2))
        for(var i=0;i<=5;i++){
            days.push(format(last_day.setDate(last_day.getDate()+1)));
        }
        
        return days;
    }

    var predicted_prices = pred_prices.map(d=>+d);
    predicted_prices.unshift(chart_stock.slice(-1).pop()["close"]);

    var trace2 = {
        x: next_days(chart_stock.slice(-1).pop()["date"]),
        y: predicted_prices,
        mode: "lines",
        name: "Predicted Prices",
        type: "scatter"
    };

    var layout = {
        title: `${code.toUpperCase()} Close Prices`,
        xaxis: {
            tickformat: "%m-%d-%Y"
        }
    };

    Plotly.newPlot("chartDiv", [trace1, trace2], layout);
}

chart()