
d3.csv("static/resources/stock_codes.csv").then(data =>{
    source = [];
    data.forEach(d => {
        if(companies.includes(d.code))
            source.push({label: `[${d.code}] ${d.name}`, value: d.code});
    });
    $('#auto').autocomplete({
        source: (request, response) => {
            var results = $.ui.autocomplete.filter(source, request.term);
            response(results.slice(0, 10));
        }
    });
});