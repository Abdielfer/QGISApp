import whitebox_workflows as wbw

wbe = wbw.WbEnvironment("frail-farming-salamander")
try:
    print(wbe.version())
except Exception as e:
  print("The error raised is: ", e)
finally:
    print(wbe.check_in_license("frail-farming-salamander"))

def checkIn():
   return wbe.check_in_license("frail-farming-salamander")