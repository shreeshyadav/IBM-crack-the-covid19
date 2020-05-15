import os
from flask import Flask,render_template,request,redirect
import final1 
app=Flask(__name__)

@app.route('/')
def index():
    
   
    return render_template('landing.html')

@app.route('/exe/',methods=['GET','POST'])
def exe():
    output=0
    if request.method=="POST":
        var1=float(request.form['var1'])
        var2=float(request.form['var2'])
        output=float(var1)+float(var2)
        output1=final1.execute(var1,var2)
        
        return render_template('index.html',outp=output,outp1=output1,link="/static/images/"+str(var1)+str(var2)+'.png')
    else: 
        return render_template('index.html',outp=output)
        

if __name__=="__main__": 
    app.run(debug=True)
