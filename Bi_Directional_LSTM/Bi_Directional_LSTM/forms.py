from django import forms 


class emailform(forms.Form):  
    text= forms.CharField(widget=forms.Textarea(attrs={'width':"90%", 'cols' : "70", 'rows': "10", })
,label="Email message",max_length=1000)  
   