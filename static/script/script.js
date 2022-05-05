async function getData(value){
    const response=await fetch('static/nutri1.json');
    const data=await response.json();
    console.log(data);
    var name=data.name[value];
    var cal=data.calories[value];
    var carb=data.carbohydrates[value];
    var pro=data.protein[value];
    var fat=data.fat[value];
    var calcium=data.calcium[value];
    var vat=data.vitamins[value];
    document.getElementById('name1').textContent=name;
    document.getElementById('cal1').textContent=cal;
    document.getElementById('carb1').textContent=carb;
    document.getElementById('pro1').textContent=pro;
    document.getElementById('fat1').textContent=fat;
    document.getElementById('calc1').textContent=calcium;
    document.getElementById('vat1').textContent=vat;
    
    

    /**const rows=data.split('\n').slice(1);
    console.log(rows);
    */
}

