import os
from openai import OpenAI
from xoxxox.shared import Custom

#---------------------------------------------------------------------------

class SenPrc:

  def __init__(self, config="xoxxox/config_sengpt_000", **dicprm):
    diccnf = Custom.update(config, dicprm)
    self.nmodel = diccnf["nmodel"]
    self.tmpmdl = diccnf["tmpmdl"]
    self.maxlen = diccnf["maxlen"]
    self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"] )

  def status(self, config="xoxxox/config_sengpt_000", **dicprm):
    diccnf = Custom.update(config, dicprm)
    self.system = diccnf["system"]
    self.prompt = diccnf["prompt"]

  def infere(self, txtreq):
    prmusr = self.prompt.format(txtreq=txtreq)
    #print(prmusr, flush=True) # DBG
    jsnans = self.client.chat.completions.create(
      model=self.nmodel,
      messages=[
        {"role": "system", "content": self.system},
        {"role": "user", "content": prmusr}
      ],
      temperature=self.tmpmdl,
      max_tokens=self.maxlen
    )
    txtres = jsnans.choices[0].message.content.strip()
    if txtres != "0" and txtres != "1":
      txtres = 0
    return txtres
